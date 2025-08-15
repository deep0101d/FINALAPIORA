// server.js — EDU AI Lab backend (ESM) — Full Fixed + Supercharged 🚀
//
// Endpoints:
//  GET   /                         -> health check
//  POST  /chat                     { question, language? }
//  POST  /summarize-text           { text, language? }
//  POST  /summarize-pdf            multipart/form-data: file=PDF, language?
//  POST  /generate-quiz            { text, count?, language? }
//  POST  /flashcards               { text, count?, language? }
//  POST  /study-planner            { subjects:[], examDate, hoursPerDay?, language? }
//  POST  /mindmap                  { text, language? }  -> returns Mermaid code
//  POST  /motivation               { context?, language? }
//
//  --- New advanced routes ---
//  POST  /essay-feedback           { essay, language? }
//  POST  /paraphrase               { text, tone?, variations?, language? }
//  POST  /tutor                    { question, language? }   (simple in-memory convo memory)
//  POST  /extract-table            { text, language? }       (returns markdown + JSON rows)
//  POST  /generate-citations       { references, style?, language? }
//  POST  /knowledge-graph          { text, language? }       (Mermaid ER diagram)
//  POST  /summarize-audio          multipart/form-data: file=audio OR { transcript }, language?
//                                  (optional Whisper CLI support via env WHISPER_CLI)

import express from "express";
import cors from "cors";
import multer from "multer";
import mammoth from "mammoth";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";
import pdfjsLib from "pdfjs-dist/legacy/build/pdf.js";
import { fileURLToPath } from "url";
import { exec } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ======= API Key (env preferred) =======
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "AIzaSyCwSIJA62axl23pdvoVrZBiesZ7HRRwHRQ";

if (!GEMINI_API_KEY || /PASTE_YOUR_KEY_HERE/i.test(GEMINI_API_KEY)) {
  console.warn("⚠️  GEMINI_API_KEY not set. Set env var GEMINI_API_KEY before running.");
}

// ======= App setup =======
const app = express();
const PORT = process.env.PORT || 30002;

app.use(cors());
app.use(express.json({ limit: "5mb" }));

// ======= File uploads =======
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir);

const upload = multer({
  dest: uploadsDir,
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB
});

// ======= Gemini (2.5 pro) =======
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// ======= Helpers =======
async function askGemini(prompt, temperature = 0.7) {
  const res = await model.generateContent({
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { temperature },
  });
  const txt = res?.response?.text ? res.response.text() : "";
  return (txt || "").trim();
}

function wrapBilingual(text, language) {
  if (!language || /^none$/i.test(language)) return text;
  return `${text}\n\n---\n\n🔁 ${language} Translation:\nTranslate the entire answer above to ${language}, preserving structure, headings, lists and tone.`;
}

async function readTxt(fp) {
  return fs.promises.readFile(fp, "utf-8");
}

async function readDocx(fp) {
  const res = await mammoth.extractRawText({ path: fp });
  return res.value || "";
}




async function readPdf(filePath) {
  const data = new Uint8Array(await fs.promises.readFile(filePath));
  const pdf = await pdfjsLib.getDocument({ data }).promise;

  let text = "";
  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();
    text += content.items.map(item => item.str).join(" ") + "\n\n";
  }
  return text;
}

export default readPdf;
// ======= Prompt builders =======
function promptSummary(text, lang = "English") {
  return `You are an academic summarizer. Create a deep, structured, readable summary in ${lang}.
Use this layout:

Title: (infer)

Executive Summary (6–10 sentences):
- What it's about
- Why it matters
- Context/scope
- Main results or conclusions

Key Concepts (bulleted, concise)

Step-by-Step Explanation (8–14 numbered steps)

Examples & Analogies (2–4)

Important Data/Formulae (if any)

Assumptions & Limitations (3–6)

Implications & Applications (3–6)

Common Pitfalls / Misconceptions (3–6)

🎯 10 High-Impact Takeaways (exactly 10 bullets)

Source:
${text}`;
}

function promptQuiz(text, count = 10) {
  return `Create ${count} multiple-choice questions from the content below.
Rules:
- Each question should test meaningful understanding.
- 4 options (A–D), one correct answer.
- Mix recall, inference, application.
- After options, give "Answer: <Letter> — one-line reason".

Format exactly:

1) Question text
A) ...
B) ...
C) ...
D) ...
Answer: <Letter> — reason

Content:
${text}`;
}

function promptFlashcards(text, count = 20) {
  return `Generate ${count} high-quality active-recall flashcards from the content.
Return lines exactly as:
Q: <question>
A: <answer>

CONTENT:
${text}`;
}

function promptMindmap(text) {
  return `Convert the following content to Mermaid mind map.
Return ONLY Mermaid code starting with "mindmap".
Use 3–4 levels of depth, short node text.

CONTENT:
${text}`;
}

function promptMotivation(context) {
  return `Write a 120–180 word motivational note for a student preparing for exams.
Tone: supportive, focused, practical (include 1 actionable tip).
Context (optional): ${context || "N/A"}`;
}

function promptEssayFeedback(essay) {
  return `You are a rigorous writing coach. Provide detailed, actionable feedback on the essay below.
Include sections: Overview (2–4 sentences), Strengths (bulleted), Areas to Improve (bulleted), Specific Line Edits (quote > suggest), Organization & Flow, Argument Quality, Evidence & Citations, Style & Tone, Grammar/Mechanics, Final Score (0–100) with a 1–2 sentence rationale.
Essay:
${essay}`;
}

function promptParaphrase(text, tone = "academic", variations = 3) {
  return `Paraphrase the text below into ${variations} distinct versions in a ${tone} tone.
Rules:
- Preserve meaning and citations if any.
- Avoid plagiarism; alter syntax and word choice.
- For each version, provide a 1-line note about what changed (e.g., simpler syntax, more formal).
Format:
## Version 1
<text>
Note: ...
## Version 2
<text>
Note: ...
## Version 3
<text>
Note: ...
TEXT:
${text}`;
}

function promptExtractTable(text) {
  return `Extract structured data from the text below.
Return:
1) A Markdown table
2) A JSON array of objects (keys = column headers)
Only include columns that are clearly inferable.
TEXT:
${text}`;
}

function promptCitations(refs, style = "APA") {
  return `Format the following references in ${style} style.
If any fields are missing, infer reasonably and mark [n.d.] or [Place unknown] as needed.
Return as a numbered list, then a plain-text bibliography block.
REFERENCES:
${refs}`;
}

function promptERGraph(text) {
  return `Create a Mermaid ER diagram (entity-relationship) from the following text.
Return ONLY Mermaid code starting with "erDiagram".
Use concise entity and relationship names. Prefer crow's foot notations expressible in Mermaid ER syntax.
TEXT:
${text}`;
}

function promptAudioSummary(transcript) {
  return `Summarize this lecture transcript with:
- 8–12 bullet key points
- 3 exam-style questions with answers
- 2 analogies/examples
- A 7-day spaced-repetition review plan
TRANSCRIPT:
${transcript}`;
}

//read PDF file and extract text

// ======= ROUTES =======

// Health
app.get("/", (_req, res) => {
  res.send("✅ EDU AI Lab backend is running.");
});

// 1) Chat
app.post("/chat", async (req, res) => {
  const { question, language } = req.body || {};
  if (!question || !question.trim()) return res.status(400).json({ error: "Missing 'question'" });
  try {
    let answer = await askGemini(question, 0.7);
    answer = wrapBilingual(answer, language);
    res.json({ answer });
  } catch (error) {
    console.error("chat_error:", error);
    res.status(500).json({ error: "chat_failed", details: error.message });
  }
});

// 2) Summarize text
app.post("/summarize-text", async (req, res) => {
  try {
    const { text, language } = req.body || {};
    if (!text || !text.trim()) return res.status(400).json({ error: "missing_text" });
    const prompt = promptSummary(text, "English");
    let answer = await askGemini(prompt, 0.5);
    answer = wrapBilingual(answer, language);
    res.json({ summary: answer });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "summarize_text_failed", details: String(e) });
  }
});

// 3) Summarize PDF
app.post("/summarize-pdf", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "no_file" });
  const { language } = req.body || {};
  try {
    const ext = (path.extname(req.file.originalname) || "").toLowerCase();
    if (ext !== ".pdf") return res.status(400).json({ error: "unsupported_type", hint: "Upload a PDF file." });

    const text = await readPdf(req.file.path);
    if (!text.trim()) return res.status(400).json({ error: "empty_pdf" });

    const prompt = promptSummary(text, "English");
    let summary = await askGemini(prompt, 0.4);
    summary = wrapBilingual(summary, language);
    res.json({ summary });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "summarize_pdf_failed", details: String(e) });
  } finally {
    try { fs.unlinkSync(req.file.path); } catch {}
  }
});

// 4) Generate Quiz
app.post("/generate-quiz", async (req, res) => {
  try {
    const { text, count = 10, language } = req.body || {};
    if (!text || !text.trim()) return res.status(400).json({ error: "missing_text" });

    const prompt = promptQuiz(text, Math.max(1, Math.min(+count || 10, 50)));
    let answer = await askGemini(prompt, 0.7);
    answer = wrapBilingual(answer, language);
    res.json({ quiz: answer });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "quiz_failed", details: String(e) });
  }
});

// 5) Flashcards
app.post("/flashcards", async (req, res) => {
  try {
    const { text, count = 20, language } = req.body || {};
    if (!text || !text.trim()) return res.status(400).json({ error: "missing_text" });

    const prompt = promptFlashcards(text, Math.max(1, Math.min(+count || 20, 100)));
    let answer = await askGemini(prompt, 0.6);
    answer = wrapBilingual(answer, language);
    res.json({ cards: answer });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "flashcards_failed", details: String(e) });
  }
});

// 6) Study Planner
app.post("/study-planner", async (req, res) => {
  try {
    const { subjects = [], examDate, hoursPerDay = 2, language } = req.body || {};
    if (!examDate || !subjects.length)
      return res.status(400).json({ error: "missing_fields", hint: "Provide subjects[] and examDate." });

    const prompt = `Create a day-by-day study plan until the exam date.
- Subjects: ${subjects.join(", ")}
- Exam Date: ${examDate}
- Hours/Day: ${hoursPerDay}

Format:
Title, Days Remaining, then a daily schedule (Date: topics, tasks, checkpoints).
Include weekly review slots, spaced repetition pointers, and 3 tips for exam week.`;

    let answer = await askGemini(prompt, 0.5);
    answer = wrapBilingual(answer, language);
    res.json({ plan: answer });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "planner_failed", details: String(e) });
  }
});

// 7) Mindmap (Mermaid)
app.post("/mindmap", async (req, res) => {
  try {
    const { text, language } = req.body || {};
    if (!text || !text.trim()) return res.status(400).json({ error: "missing_text" });

    const prompt = promptMindmap(text);
    let code = await askGemini(prompt, 0.5);
    code = wrapBilingual(code, language);
    res.json({ mermaid: code });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "mindmap_failed", details: String(e) });
  }
});

// 8) Motivation
app.post("/motivation", async (req, res) => {
  try {
    const { context, language } = req.body || {};
    const prompt = promptMotivation(context);
    let answer = await askGemini(prompt, 0.8);
    answer = wrapBilingual(answer, language);
    res.json({ message: answer });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "motivation_failed", details: String(e) });
  }
});

/* ===== New Advanced Features ===== */

// 9) Essay feedback
app.post("/essay-feedback", async (req, res) => {
  const { essay, language } = req.body || {};
  if (!essay || !essay.trim()) return res.status(400).json({ error: "missing_essay" });
  try {
    const prompt = promptEssayFeedback(essay);
    let answer = await askGemini(prompt, 0.7);
    res.json({ feedback: wrapBilingual(answer, language) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "essay_feedback_failed", details: String(e) });
  }
});

// 10) Paraphrase
app.post("/paraphrase", async (req, res) => {
  const { text, tone = "academic", variations = 3, language } = req.body || {};
  if (!text || !text.trim()) return res.status(400).json({ error: "missing_text" });
  try {
    const prompt = promptParaphrase(text, String(tone || "academic"), Math.max(1, Math.min(+variations || 3, 6)));
    let answer = await askGemini(prompt, 0.7);
    res.json({ paraphrases: wrapBilingual(answer, language) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "paraphrase_failed", details: String(e) });
  }
});

// 11) Tutor (simple in-memory conversation)
const tutorMemory = [];
app.post("/tutor", async (req, res) => {
  const { question, language } = req.body || {};
  if (!question || !question.trim()) return res.status(400).json({ error: "missing_question" });
  try {
    tutorMemory.push({ role: "user", text: question });
    const convo = tutorMemory.map(m => `${m.role.toUpperCase()}: ${m.text}`).join("\n");
    const prompt = `You are a patient subject-matter tutor. Continue the conversation. Use Socratic questioning and examples. Keep answers concise but helpful.
Conversation so far:
${convo}`;
    let answer = await askGemini(prompt, 0.7);
    tutorMemory.push({ role: "assistant", text: answer });
    res.json({ answer: wrapBilingual(answer, language) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "tutor_failed", details: String(e) });
  }
});

// 12) Extract table (markdown + JSON)
app.post("/extract-table", async (req, res) => {
  const { text, language } = req.body || {};
  if (!text || !text.trim()) return res.status(400).json({ error: "missing_text" });
  try {
    const prompt = promptExtractTable(text);
    let answer = await askGemini(prompt, 0.5);
    res.json({ table: wrapBilingual(answer, language) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "table_failed", details: String(e) });
  }
});

// 13) Generate citations
// helper: normalize refs input (array|string) -> string
function normalizeRefs(input) {
  if (Array.isArray(input)) {
    // filter out empties and join lines
    return input
      .map(r => (typeof r === "string" ? r : JSON.stringify(r)))
      .map(s => s.trim())
      .filter(Boolean)
      .join("\n");
  }
  if (typeof input === "string") return input.trim();
  // anything else (object/null/number) -> stringify minimally
  return String(input ?? "").trim();
}

// 13) Generate citations (robust)
app.post("/generate-citations", async (req, res) => {
  try {
    const { references, style = "APA", language } = req.body || {};
    const refsText = normalizeRefs(references);

    if (!refsText) {
      return res.status(400).json({
        error: "missing_references",
        hint: "Send `references` as a string OR array of strings."
      });
    }

    // basic size guard to avoid model blowups
    if (refsText.length > 100_000) {
      return res.status(413).json({
        error: "payload_too_large",
        hint: "Keep total references under ~100k chars."
      });
    }

    const prompt = promptCitations(refsText, style);
    let answer = await askGemini(prompt, 0.4);
    return res.json({ citations: wrapBilingual(answer, language) });
  } catch (e) {
    console.error("generate-citations_error:", e);
    // never crash: always respond
    return res.status(500).json({
      error: "citations_failed",
      details: String(e?.message || e)
    });
  }
});

// 14) Knowledge graph (Mermaid ER)
app.post("/knowledge-graph", async (req, res) => {
  const { text, language } = req.body || {};
  if (!text || !text.trim()) return res.status(400).json({ error: "missing_text" });
  try {
    const prompt = promptERGraph(text);
    let graph = await askGemini(prompt, 0.5);
    res.json({ mermaid: wrapBilingual(graph, language) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "graph_failed", details: String(e) });
  }
});



// ======= START SERVER =======
app.listen(PORT, () => {
  console.log(`✅ EDU AI Lab backend listening on http://localhost:${PORT}`);
});
