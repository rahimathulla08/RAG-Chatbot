from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from query import load_vectorstore, get_retriever, get_llm, ask_question
import webbrowser
import threading

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# Load RAG components once
# ---------------------------------------------------
print("Loading RAG components...")
vectorstore = load_vectorstore()
retriever = get_retriever(vectorstore)
llm = get_llm()
print("RAG chatbot is ready!")

# ---------------------------------------------------
# HTML + CSS + JS in one file
# ---------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>College Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            background: #f5f7fb;
            min-height: 100vh;
        }

        .page-content {
            padding: 40px;
        }

        .page-content h1 {
            color: #1e3a8a;
            margin-bottom: 10px;
            font-size: 38px;
        }

        .page-content p {
            color: #475569;
            max-width: 800px;
            line-height: 1.6;
            font-size: 17px;
        }

        .section {
            margin-top: 30px;
        }

        .section h2 {
            color: #0f172a;
            margin-bottom: 12px;
        }

        .section ul {
            padding-left: 20px;
            color: #334155;
            line-height: 1.8;
            font-size: 16px;
        }

        /* Floating bubble */
        .chat-bubble {
            position: fixed;
            bottom: 25px;
            right: 25px;
            width: 72px;
            height: 72px;
            border-radius: 50%;
            border: none;
            background: #1d4ed8;
            color: white;
            font-size: 30px;
            cursor: pointer;
            box-shadow: 0 12px 28px rgba(0,0,0,0.25);
            z-index: 1000;
            animation: bounce 2s infinite;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            25% { transform: translateY(-4px); }
            50% { transform: translateY(0); }
            75% { transform: translateY(-2px); }
        }

        /* Chat widget */
        .chat-widget {
            position: fixed;
            bottom: 50px;
            right: 25px;
            width: 390px;
            height: 580px;
            background: #ffffff;
            border-radius: 28px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.20);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            z-index: 1001;
            border: 1px solid #e5e7eb;
        }

        .hidden {
            display: none;
        }

        /* Header */
        .chat-header {
            background: #f8fafc;
            padding: 16px 18px;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .bot-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: #e0e7ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }

        .header-text h3 {
            font-size: 26px;
            color: #111827;
            margin-bottom: 2px;
        }

        .header-text p {
            font-size: 14px;
            color: #6b7280;
        }

        .close-btn {
            border: none;
            background: transparent;
            font-size: 28px;
            cursor: pointer;
            color: #6b7280;
            line-height: 1;
        }

        /* Messages area */
        .chat-messages {
            flex: 1;
            padding: 18px;
            overflow-y: auto;
            background: #ffffff;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        /* Welcome card */
        .welcome-card {
            background: #f3f4f6;
            border-radius: 40px;
            padding: 18px;
            color: #111827;
            line-height: 1.6;
            font-size: 15px;
            max-width: 92%;
        }

        .welcome-meta {
            font-size: 13px;
            color: #6b7280;
            margin-top: 8px;
            margin-left: 4px;
        }

        /* Message bubbles */
        .bot-message,
        .user-message {
            max-width: 82%;
            padding: 14px 16px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.6;
            word-wrap: break-word;
            white-space: pre-wrap;
        }

        .bot-message {
            background: #f3f4f6;
            color: #111827;
            align-self: flex-start;
            border-bottom-left-radius: 6px;
        }

        .user-message {
            background: #2563eb;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 6px;
        }

        /* Input area */
        .chat-input-wrapper {
            padding: 16px;
            border-top: 1px solid #e5e7eb;
            background: white;
        }

        .chat-input-box {
            border: 2px solid #2563eb;
            border-radius: 24px;
            padding: 14px 14px 12px 14px;
            background: #fff;
        }

        .email-line {
            width: 100%;
            border: none;
            outline: none;
            font-size: 15px;
            color: #6b7280;
            margin-bottom: 10px;
            background: transparent;
        }

        .divider {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 6px 0 12px 0;
        }

        .question-row {
            display: flex;
            align-items: flex-end;
            gap: 10px;
        }

        .question-input {
            flex: 1;
            border: none;
            outline: none;
            resize: none;
            font-size: 16px;
            min-height: 70px;
            max-height: 140px;
            line-height: 1.5;
            color: #111827;
        }

        .question-input::placeholder {
            color: #6b7280;
        }

        .send-btn {
            width: 46px;
            height: 46px;
            border: none;
            border-radius: 50%;
            background: #2563eb;
            color: white;
            font-size: 20px;
            cursor: pointer;
            flex-shrink: 0;
        }

        .send-btn:disabled {
            background: #cbd5e1;
            cursor: not-allowed;
        }

        /* Typing indicator */
        .typing {
            background: #f3f4f6;
            color: #6b7280;
            align-self: flex-start;
            border-bottom-left-radius: 6px;
            max-width: 120px;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 14px;
        }

        /* Mobile responsive */
        @media (max-width: 520px) {
            .chat-widget {
                width: calc(100vw - 20px);
                right: 10px;
                bottom: 95px;
                height: 80vh;
            }

            .chat-bubble {
                right: 16px;
                bottom: 16px;
            }

            .page-content {
                padding: 20px;
            }

            .page-content h1 {
                font-size: 30px;
            }
        }
    </style>
</head>
<body>

    <div class="page-content">
        <h1>🎓 College Campus Portal</h1>
        <p>
            Welcome to the student academic portal. Ask questions about exam rules, attendance, AIML syllabus,
            placement policy, and internship guidelines using the chatbot at the bottom-right.
        </p>

        <div class="section">
            <h2>Student Services</h2>
            <ul>
                <li>Exam rules and hall ticket instructions</li>
                <li>AIML syllabus information</li>
                <li>Attendance requirements</li>
                <li>Placement eligibility and process</li>
                <li>Internship guidelines and submission rules</li>
            </ul>
        </div>
    </div>

    <!-- Floating Bubble -->
    <button id="chatBubble" class="chat-bubble">💬</button>
    </button>

    <!-- Chat Widget -->
    <div id="chatWidget" class="chat-widget hidden">
        <div class="chat-header">
            <div class="header-left">
                <div class="bot-avatar">🎓</div>
                <div class="header-text">
                    <h3>CollegeBot</h3>
                    <p>College academic information assistant</p>
                </div>
            </div>
            <button id="closeChat" class="close-btn">×</button>
        </div>

        <div id="chatMessages" class="chat-messages">
            <div class="welcome-card">
                Hi I'm your college AI assistant. Ask about exams, syllabus, placements & internships:
            </div>
            <div class="welcome-meta">CollegeBot Just now</div>
        </div>

        <div class="chat-input-wrapper">
                <div class="question-row">
                    <textarea id="userInput" class="question-input" placeholder="Ask a question..."></textarea>
                    <button id="sendBtn" class="send-btn">↑</button>
                </div>
        </div>
    </div>

    <script>
        const chatBubble = document.getElementById("chatBubble");
        const chatWidget = document.getElementById("chatWidget");
        const closeChat = document.getElementById("closeChat");
        const sendBtn = document.getElementById("sendBtn");
        const userInput = document.getElementById("userInput");
        const chatMessages = document.getElementById("chatMessages");

        // Open chat
        chatBubble.addEventListener("click", () => {
            chatWidget.classList.remove("hidden");
            chatBubble.style.display = "none";
            userInput.focus();
        });

        // Close chat
        closeChat.addEventListener("click", () => {
            chatWidget.classList.add("hidden");
            chatBubble.style.display = "block";
        });

        function addMessage(text, sender) {
            const msg = document.createElement("div");
            msg.classList.add(sender === "user" ? "user-message" : "bot-message");
            msg.textContent = text;
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function addTyping() {
            const typing = document.createElement("div");
            typing.classList.add("typing");
            typing.id = "typing-indicator";
            typing.textContent = "Typing...";
            chatMessages.appendChild(typing);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function removeTyping() {
            const typing = document.getElementById("typing-indicator");
            if (typing) typing.remove();
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage(message, "user");
            userInput.value = "";
            userInput.style.height = "70px";

            addTyping();

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();
                removeTyping();
                addMessage(data.answer || "I couldn't generate an answer.", "bot");
            } catch (error) {
                removeTyping();
                addMessage("Error connecting to the chatbot backend.", "bot");
                console.error(error);
            }
        }

        sendBtn.addEventListener("click", sendMessage);

        userInput.addEventListener("keydown", function(e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-grow textarea
        userInput.addEventListener("input", function() {
            this.style.height = "70px";
            this.style.height = Math.min(this.scrollHeight, 140) + "px";
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------
# Home page route
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


# ---------------------------------------------------
# Chat API route
# ---------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"answer": "Please enter a question."})

    try:
        answer, _ = ask_question(user_message, retriever, llm)
        return jsonify({"answer": answer})
    except Exception as e:
        print("Chat error:", e)
        return jsonify({"answer": "Something went wrong while generating the answer."})


# ---------------------------------------------------
# Run app
# ---------------------------------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)
