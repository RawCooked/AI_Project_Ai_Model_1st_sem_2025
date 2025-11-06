import os
import json
import queue
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from dotenv import load_dotenv

# LangChain (functional/chain-based)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
try:
    from langchain_community.chat_models import ChatOpenAI  # older/newer community driver
except Exception:
    from langchain_openai import ChatOpenAI  # fallback to official package

# External utils used by tools
import requests
import wikipedia

load_dotenv()

# =========================
# CONFIG (module-level, not OOP)
# =========================
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", 0.01))
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", 2.0))
MAX_RECORDING_TIME = int(os.getenv("MAX_RECORDING_TIME", 30))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

ESPRIT_API_KEY = os.getenv("ESPRIT_API_KEY")
ESPRIT_BASE_URL = os.getenv("ESPRIT_BASE_URL", "https://tokenfactory.esprit.tn/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "hosted_vllm/Llama-3.1-70B-Instruct")

# =========================
# VOICE CAPTURE (functional)
# =========================
_audio_queue = queue.Queue()

def _audio_callback(indata, frames, time, status):
    if status:
        print(status)
    _audio_queue.put(indata.copy())


def record_until_silence(sample_rate=SAMPLE_RATE, silence_threshold=SILENCE_THRESHOLD,
                          silence_duration=SILENCE_DURATION, max_duration=MAX_RECORDING_TIME):
    print("🎤 Listening...")
    recorded_chunks = []
    silence_chunks = 0
    silence_threshold_chunks = int(silence_duration * sample_rate / 1024)

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=_audio_callback, blocksize=1024):
        start = datetime.now()
        speech_detected = False

        while True:
            if (datetime.now() - start).total_seconds() > max_duration:
                print("⏱️ Max time reached")
                break
            try:
                chunk = _audio_queue.get(timeout=0.1)
                recorded_chunks.append(chunk)
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                if energy > silence_threshold:
                    silence_chunks = 0
                    if not speech_detected:
                        print("🗣️ Speech detected...")
                        speech_detected = True
                else:
                    if speech_detected:
                        silence_chunks += 1
                if speech_detected and silence_chunks > silence_threshold_chunks:
                    print("🤫 Processing...")
                    break
            except queue.Empty:
                continue
    if not recorded_chunks:
        return None
    return np.concatenate(recorded_chunks, axis=0)


def save_audio(audio_data, filename="temp_audio.wav", sample_rate=SAMPLE_RATE):
    audio_int16 = np.int16(audio_data * 32767)
    write(filename, sample_rate, audio_int16)
    return filename


def transcribe_audio(filename, whisper_model):
    print("🧠 Transcribing...")
    result = whisper_model.transcribe(filename, language="en")
    return result.get("text", "").strip()

# =========================
# TOOLS (LangChain Tool API)
# =========================

def tool_search_web(query: str) -> str:
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(search_url, headers=headers, timeout=10)
        from bs4 import BeautifulSoup  # lightweight and common
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        for result in soup.find_all('a', class_='result__snippet', limit=3):
            text = result.get_text(strip=True)
            if text and len(text) > 20:
                results.append(f"• {text}")
        if results:
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        return f"Limited information found for '{query}'. Try rephrasing."
    except Exception as e:
        return f"Search unavailable. Error: {str(e)}"


def tool_wikipedia(topic: str) -> str:
    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=4, auto_suggest=True)
        page = wikipedia.page(topic, auto_suggest=True)
        return f"Wikipedia - {page.title}:\n\n{summary}\n\nURL: {page.url}"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"


def tool_calculate(expression: str) -> str:
    try:
        allowed = set('0123456789+-*/(). ')
        if not all(c in allowed for c in expression):
            return "Invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def tool_now() -> str:
    now = datetime.now()
    return f"Date: {now.strftime('%A, %B %d, %Y')}\nTime: {now.strftime('%I:%M:%S %p')}"


def tool_note(args: str) -> str:
    try:
        # expects "title, content"
        parts = [p.strip() for p in args.split(',', 1)]
        title = parts[0] if parts else "Note"
        content = parts[1] if len(parts) > 1 else ""
        filename = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{title.replace(' ', '_')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}")
        return f"✅ Note saved to {filename}"
    except Exception as e:
        return f"Error: {str(e)}"


def tool_save(content: str) -> str:
    try:
        filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ Saved to: {os.path.abspath(filename)}"
    except Exception as e:
        return f"Error: {str(e)}"


def tool_create_prescription(patient_info: str, medications: str, diagnosis: str = "", notes: str = "") -> str:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT

        patient_parts = [p.strip() for p in patient_info.split(',')]
        patient_name = patient_parts[0] if len(patient_parts) > 0 else "Patient"
        patient_age = patient_parts[1] if len(patient_parts) > 1 else "N/A"
        patient_gender = patient_parts[2] if len(patient_parts) > 2 else "N/A"

        med_list = []
        if medications:
            for med in medications.split(';'):
                med = med.strip()
                if med:
                    parts = med.split(':')
                    med_list.append({
                        'name': parts[0].strip() if len(parts) > 0 else '',
                        'dosage': parts[1].strip() if len(parts) > 1 else '',
                        'frequency': parts[2].strip() if len(parts) > 2 else '',
                        'duration': parts[3].strip() if len(parts) > 3 else ''
                    })

        clean_name = patient_name.replace(' ', '_').replace('"', '').replace("'", '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prescription_{clean_name}_{timestamp}.pdf"

        doc = SimpleDocTemplate(filename, pagesize=letter,
                                 rightMargin=0.75*inch, leftMargin=0.75*inch,
                                 topMargin=0.75*inch, bottomMargin=0.75*inch)
        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                     fontSize=24, textColor=colors.HexColor('#1a5490'), spaceAfter=30,
                                     alignment=TA_CENTER, fontName='Helvetica-Bold')
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
                                       fontSize=14, textColor=colors.HexColor('#2c5aa0'), spaceAfter=12,
                                       spaceBefore=12, fontName='Helvetica-Bold')
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'],
                                      fontSize=11, spaceAfter=6)

        elements.append(Paragraph("℞ MEDICAL PRESCRIPTION", title_style))
        elements.append(Spacer(1, 0.2*inch))

        doctor_data = [
            ['Dr. AI Medical Assistant', ''],
            ['Medical AI System', 'Date: ' + datetime.now().strftime("%B %d, %Y")],
            ['License: AI-ASSISTANT-001', 'Time: ' + datetime.now().strftime("%I:%M %p")]
        ]
        doctor_table = Table(doctor_data, colWidths=[3.5*inch, 3*inch])
        doctor_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        elements.append(doctor_table)
        elements.append(Spacer(1, 0.2*inch))

        elements.append(Paragraph("PATIENT INFORMATION", heading_style))
        patient_data = [
            ['Name:', patient_name, 'Age:', patient_age],
            ['Gender:', patient_gender, 'Date:', datetime.now().strftime("%m/%d/%Y")]
        ]
        patient_table = Table(patient_data, colWidths=[1*inch, 2.5*inch, 0.8*inch, 2.2*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.2*inch))

        elements.append(Paragraph("PRESCRIPTION", heading_style))
        if med_list:
            med_data = [['#', 'Medication', 'Dosage', 'Frequency', 'Duration']]
            for idx, med in enumerate(med_list, 1):
                med_data.append([str(idx), med['name'], med['dosage'], med['frequency'], med['duration']])
            med_table = Table(med_data, colWidths=[0.4*inch, 2.2*inch, 1.3*inch, 1.5*inch, 1.1*inch])
            med_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ]))
            elements.append(med_table)

        if notes:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("INSTRUCTIONS & NOTES", heading_style))
            elements.append(Paragraph(notes, normal_style))

        elements.append(Spacer(1, 0.3*inch))
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
                                          fontSize=8, textColor=colors.HexColor('#666666'), alignment=TA_CENTER)
        elements.append(Paragraph("⚠️ AI-generated prescription for informational purposes only. Consult a licensed healthcare professional.", disclaimer_style))
        elements.append(Spacer(1, 0.4*inch))
        sig_style = ParagraphStyle('Signature', parent=styles['Normal'], fontSize=10, alignment=TA_RIGHT)
        elements.append(Paragraph("_____________________________", sig_style))
        elements.append(Paragraph("AI Medical Assistant", sig_style))

        doc.build(elements)
        return f"✅ Prescription created!\n📄 {os.path.abspath(filename)}\n👤 {patient_name}\n💊 {len(med_list)} medications"
    except ImportError:
        return "❌ Install reportlab: pip install reportlab"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def tool_quick_prescription(args: str) -> str:
    try:
        # expects: "Patient Name, condition"
        parts = [p.strip() for p in args.split(',', 1)]
        patient_name = parts[0] if parts else "Patient"
        condition = parts[1] if len(parts) > 1 else ""
        presets = {
            'common cold': ('Acetaminophen:500mg:Every 6 hours:5 days; Vitamin C:1000mg:Once daily:7 days',
                            'Rest, drink 8-10 glasses fluids daily. Return if fever >102°F.'),
            'flu': ('Oseltamivir:75mg:Twice daily:5 days; Acetaminophen:500mg:Every 6 hours:7 days',
                    'Isolation 5 days. Rest and fluids. Return if breathing difficulties.'),
            'headache': ('Ibuprofen:400mg:Every 8 hours as needed:3 days', 'Avoid bright lights. Stay hydrated.'),
            'fever': ('Acetaminophen:500mg:Every 6 hours:5 days', 'Monitor temp every 4 hours. Return if >103°F.'),
            'cough': ('Dextromethorphan:10ml:Every 6 hours:5 days; Ambroxol:30mg:3x daily:7 days', 'Stay hydrated. Use humidifier. Avoid smoking.'),
            'sore throat': ('Amoxicillin:500mg:3x daily:7 days; Throat lozenges:As needed:Every 3-4h:7 days', 'Warm salt water gargles. Complete antibiotic course.'),
            'allergies': ('Cetirizine:10mg:Once daily:14 days; Nasal spray:2 sprays:Once daily:14 days', 'Avoid allergens. May cause drowsiness.'),
            'back pain': ('Ibuprofen:600mg:3x daily with food:7 days; Muscle relaxant:10mg:Bedtime:5 days', 'Hot/cold packs. Gentle stretching. Avoid heavy lifting.'),
            'stomach pain': ('Omeprazole:20mg:Before breakfast:14 days; Antacid:As directed:As needed:14 days', 'Small frequent meals. Avoid spicy/fatty foods.'),
        }
        key = condition.lower().strip()
        matched = next((k for k in presets if k in key or key in k), None)
        if matched:
            meds, notes = presets[matched]
            diagnosis = matched.title()
        else:
            meds = "General medication:As directed:As needed:7 days"
            notes = "Follow instructions. Return if symptoms worsen."
            diagnosis = condition.title() if condition else "General"
        return tool_create_prescription(f"{patient_name}, N/A, N/A", meds, diagnosis, notes)
    except Exception as e:
        return f"❌ Error: {str(e)}"


TOOLS = [
    Tool.from_function(func=tool_search_web, name="search_web", description="Search the web for short, recent snippets. Input: search query string."),
    Tool.from_function(func=tool_wikipedia, name="get_wikipedia_info", description="Get a short summary + URL for a topic from Wikipedia. Input: topic string."),
    Tool.from_function(func=tool_calculate, name="calculate", description="Calculate basic math expressions. Input: expression string."),
    Tool.from_function(func=tool_now, name="get_current_datetime", description="Current date and time. No input."),
    Tool.from_function(func=tool_note, name="create_note", description="Create a quick note file. Input: 'title, content'"),
    Tool.from_function(func=tool_save, name="save_to_file", description="Save content to a timestamped text file. Input: content string."),
    Tool.from_function(func=tool_create_prescription, name="create_prescription", description="Create a prescription PDF. Input: patient_info, medications, [diagnosis], [notes]"),
    Tool.from_function(func=tool_quick_prescription, name="quick_prescription", description="Quick prescription for common conditions. Input: 'Patient Name, condition'"),
]

# =========================
# LANGCHAIN LLM AND CHAIN
# =========================


def build_llm() -> ChatOpenAI:
    if not ESPRIT_API_KEY:
        raise RuntimeError("ESPRIT_API_KEY not found in .env")
    llm = ChatOpenAI(
        api_key=ESPRIT_API_KEY,
        base_url=ESPRIT_BASE_URL,
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=500,
    )
    return llm


SYSTEM_PROMPT = (
    "You are a helpful AI voice assistant with access to tools. Keep responses natural, "
    "conversational, and concise (2-3 sentences unless more detail is requested).\n\n"
    "Tool usage policy:\n"
    "- Prefer using tools when needed.\n"
    "- For medical intents, ensure the patient's real name is known. If not, ask for it first.\n"
    "- After using a tool, summarize results naturally.\n"
    "Safety: Medical outputs are informational only; advise consulting a licensed professional."
)


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])


parser = StrOutputParser()


# Simple ReAct-like loop using LangChain tools programmatically (functional)

def run_agent(user_input: str, llm: ChatOpenAI) -> str:
    initial = (prompt | llm | parser).invoke({"input": user_input})

    def parse_tool_call(text: str) -> Optional[tuple]:
        if "TOOL:" in text and "ARGS:" in text:
            lines = [l.strip() for l in text.strip().splitlines()]
            tool = None
            args = None
            for l in lines:
                if l.startswith("TOOL:"):
                    tool = l.replace("TOOL:", "").strip()
                if l.startswith("ARGS:"):
                    args = l.replace("ARGS:", "").strip()
            if tool:
                return tool, args or ""
        return None

    tool_call = parse_tool_call(initial)
    if not tool_call:
        return initial

    name, args = tool_call
    tool_map = {t.name: t for t in TOOLS}
    if name not in tool_map:
        return f"(Tool '{name}' not found)\n\n{initial}"

    try:
        result = tool_map[name].invoke(args)
    except TypeError:
        result = tool_map[name].invoke("")

    followup_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", f"User asked: {{orig}}\n\nTool used: {name}\nTool result:\n{{tool_result}}\n\nProvide a natural answer to the user.")
    ])
    final = (followup_template | llm | parser).invoke({"orig": user_input, "tool_result": result})
    return final


# =========================
# MAIN LOOP (functional)
# =========================

def main():
    print("=" * 60)
    print("🎙️  AI VOICE AGENT WITH PRESCRIPTION SYSTEM (LangChain)")
    print("=" * 60)
    print("✨ Tools: Search, Wikipedia, Calculate, Save, Notes, Prescriptions")
    print("=" * 60)
    if not ESPRIT_API_KEY:
        print("❌ ESPRIT_API_KEY not found in .env!")
        return

    print("🔄 Loading Whisper...")
    whisper_model = whisper.load_model(WHISPER_MODEL)

    print("🔄 Initializing LLM...")
    llm = build_llm()

    print("✅ Ready!  Press Ctrl+C to exit.\n")
    print("💡 Try:")
    print("   - 'My name is [Your Name] and I have a headache'")
    print("   - 'What time is it?'")
    print("   - 'Search for AI news'")
    print("   - 'Create prescription for John with cold'\n")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_history = []

    turn = 1
    try:
        while True:
            print(f"\n{'─' * 60}")
            print(f"Turn {turn}")
            print(f"{'─' * 60}")

            audio = record_until_silence()
            if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                print("⚠️ No speech, trying again...")
                continue

            audio_file = save_audio(audio)
            user_text = transcribe_audio(audio_file, whisper_model)
            if not user_text or len(user_text.strip()) < 2:
                print("⚠️ Could not understand...")
                continue

            print(f"\n👤 You: {user_text}")
            ai_text = run_agent(user_text, llm)
            print(f"\n🤖 AI: {ai_text}")

            full_history.append({
                "user": user_text,
                "assistant": ai_text,
                "timestamp": datetime.now().isoformat(),
            })
            turn += 1
    except KeyboardInterrupt:
        print("\n\n👋 Ending...")
        data = {
            "session_id": session_id,
            "history": full_history,
            "total_turns": len(full_history),
        }
        fn = f"conversation_{session_id}.json"
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {fn}")
        print("Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        data = {
            "session_id": session_id,
            "history": full_history,
            "total_turns": len(full_history),
            "error": str(e)
        }
        fn = f"conversation_{session_id}.json"
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
