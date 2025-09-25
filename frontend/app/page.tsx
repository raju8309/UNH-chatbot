"use client";


import React, { useState, useRef, useEffect } from "react";
// API endpoint for chat
const CHAT_API_URL = "/chat";

export type ChatMessage = {
  role: string;
  content: string;
  sources?: string[];
};

export default function Home() {
    // Chat state and logic
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const chatEndRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const sendMessage = async (e: React.FormEvent) => {
      e.preventDefault();
      if (!input.trim()) return;

      setMessages(prev => [...prev, { role: "user", content: input }]);
      const userInput = input;
      setInput("");

      try {
        const res = await fetch(CHAT_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: userInput }),
        });
        if (!res.ok) throw new Error("API error");
        const data = await res.json();
        setMessages(prev => [...prev, { role: "bot", content: data.answer, sources: data.sources }]);
      } catch (err) {
        setMessages(prev => [...prev, { role: "bot", content: "Sorry, there was an error connecting to the chatbot API.", sources: [] }]);
      }
    };

    // Suggested questions for cards
    const suggestedQuestions = [
      "What is the time limit for a master’s degree?",
      "How many thesis credits must a master’s student enroll in?",
      "Are Ph.D. students assigned a guidance committee?",
      "How many courses are required for a graduate certificate program?",
    ];

    // When a card is clicked, immediately send the question
    const handleCardClick = async (q: string) => {
      setMessages(prev => [...prev, { role: "user", content: q }]);
      try {
        const res = await fetch(CHAT_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: q }),
        });
        if (!res.ok) throw new Error("API error");
        const data = await res.json();
        setMessages(prev => [...prev, { role: "bot", content: data.answer, sources: data.sources }]);
      } catch (err) {
        setMessages(prev => [...prev, { role: "bot", content: "Sorry, there was an error connecting to the chatbot API.", sources: [] }]);
      }
    };

    return (
      <main className="h-screen flex flex-col bg-[var(--unh-white)] overflow-hidden">
        <header className="bg-[var(--unh-blue)] px-8 py-4 text-center shadow-md" style={{ color: '#fff' }}>
          <div className="flex items-center">
            <img src="/unh.svg" alt="UNH Logo" className="my-6 mr-4" style={{ maxWidth: '125px', height: 'auto', width: 'auto', marginTop: '24px', marginBottom: '24px' }} />
            <span className="text-3xl font-bold" style={{ fontFamily: 'Glypha, Arial, sans-serif' }}>Graduate Catalog Chatbot</span>
          </div>
        </header>
        {/* Suggested questions section, hidden after user asks a question */}
        {messages.find(m => m.role === "user") === undefined && (
          <section className="flex flex-col justify-center items-center w-full max-w-3xl mx-auto min-h-[60vh] pt-24 pb-2" style={{flex: 1}}>
            <h1 className="text-3xl font-bold mb-6 text-center text-gray-400">What can I help with?</h1>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full">
              {suggestedQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleCardClick(q)}
                  className="bg-white shadow-md rounded-xl p-6 text-lg font-medium hover:bg-[var(--unh-light-gray)] hover:shadow-lg transition flex items-center justify-center min-h-[120px]"
                >
                  {q}
                </button>
              ))}
            </div>
          </section>
        )}
        <div className="flex-1 flex flex-col items-center overflow-hidden">
          <div className="w-2/3 flex flex-col h-full">
            <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-2 scrollbar-hide" style={{wordBreak: 'break-word'}}>
              {messages.map((msg, i) => {
                const isRoleChange = i > 0 && messages[i - 1].role !== msg.role;
                if (msg.role === "user") {
                  return (
                    <div
                      key={i}
                      className={`flex justify-end mb-2${i === messages.length - 1 ? ' mb-6' : ''}`}
                      style={isRoleChange ? { marginTop: '1rem' } : {}}
                    >
                      <div className="bg-[var(--unh-blue)] text-[var(--unh-white)] rounded-2xl break-words whitespace-pre-line m-1 px-6 py-4 text-lg md:text-xl max-w-[800px] w-fit block box-border">
                        {msg.content}
                      </div>
                    </div>
                  );
                } else {
                  return (
                    <div
                      key={i}
                      className={`flex justify-start mb-2${i === messages.length - 1 ? ' mb-6' : ''}`}
                      style={isRoleChange ? { marginTop: '1rem' } : {}}
                    >
                      <div className="bg-[var(--unh-light-gray)] text-black rounded-2xl break-words whitespace-pre-line m-1 px-6 py-4 text-lg md:text-xl max-w-[800px] w-fit block box-border">
                        <div>{msg.content}</div>
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="mt-4">
                            <div className="font-semibold text-sm mb-1">Sources:</div>
                            <ul className="list-disc list-inside text-sm text-gray-700">
                              {msg.sources.map((src, idx) => {
                                // Parse format: "- Title (url)" or "- Title"
                                const match = src.match(/^-\s*(.+)\s*\(([^)]+)\)$/);
                                if (match) {
                                  const title = match[1];
                                  const url = match[2];
                                  return (
                                    <li key={idx}>
                                      {url ? (
                                        <a href={url} target="_blank" rel="noopener noreferrer" className="underline text-blue-700">
                                          {title}
                                        </a>
                                      ) : (
                                        title
                                      )}
                                    </li>
                                  );
                                } else {
                                  // Fallback for unexpected format
                                  return <li key={idx}>{src}</li>;
                                }
                              })}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                }
              })}
              <div ref={chatEndRef} />
            </div>
            <form onSubmit={sendMessage} className="flex gap-2 bg-white py-2 px-4">
              <div className="flex-1 flex items-center">
                <div className="relative w-full flex items-center">
                  <input
                    className="w-full rounded-full border-2 border-gray-400 text-black pr-14 px-8 py-6 text-lg md:text-xl placeholder:text-gray-400 bg-transparent box-border focus:outline-none"
                    type="text"
                    placeholder="Ask questions about programs, courses, and policies"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                  />
                  <button
                    type="submit"
                    className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-[var(--unh-blue)] p-3 flex items-center justify-center shadow hover:bg-[var(--unh-accent-blue)] transition-colors"
                    aria-label="Send"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="white" className="w-6 h-6">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              </div>
            </form>
            <div className="w-full px-4 pb-2 text-center text-sm text-gray-500">
              <p className="text-lg">Not answering your question? Contact us at grad.school@unh.edu.</p>
            </div>
          </div>
        </div>
      </main>
    );
}
