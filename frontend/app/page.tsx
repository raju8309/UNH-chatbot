"use client";

import React, { useState, useRef, useEffect } from "react";

const CHAT_API_URL = "/chat";

export type ChatMessage = {
  role: string;
  content: string;
  sources?: string[];
  hasAlternative?: boolean;
  alternativeAnswer?: string;
  alternativeSources?: string[];
  answerMode?: string;
  goldSimilarity?: number;
  selectedVersion?: "primary" | "alternative";
};

function linkify(text: string) {
  let out = text;
  const mdLinkRx = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
  if (mdLinkRx.test(out)) {
    out = out.replace(
      mdLinkRx,
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-700 underline">$1</a>'
    );
  } else {
    out = out.replace(
      /(https?:\/\/[^\s)]+)(\)?)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-blue-700 underline">$1</a>'
    );
  }
  return out;
}

function AnswerVersion({ 
  answer, 
  sources, 
  isSelected, 
  onSelect, 
  label, 
  badge 
}: {
  answer: string;
  sources?: string[];
  isSelected: boolean;
  onSelect: () => void;
  label: string;
  badge?: string;
}) {
  return (
    <div 
      className={`rounded-2xl p-4 cursor-pointer transition-all ${
        isSelected 
          ? 'bg-[var(--unh-light-gray)] shadow-md ring-2 ring-[var(--unh-blue)]' 
          : 'bg-gray-50 hover:bg-gray-100'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
            isSelected ? 'border-[var(--unh-blue)] bg-[var(--unh-blue)]' : 'border-gray-400'
          }`}>
            {isSelected && <div className="w-2 h-2 bg-white rounded-full"></div>}
          </div>
          <span className="font-medium text-sm">{label}</span>
        </div>
        {badge && (
          <span className="text-xs px-2 py-1 rounded bg-blue-100 text-blue-800 font-medium">
            {badge}
          </span>
        )}
      </div>
      
      <div 
        className="text-black text-base"
        dangerouslySetInnerHTML={{ __html: linkify(answer) }}
      />
      
      {sources && sources.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-300">
          <div className="font-semibold text-sm mb-1">Sources:</div>
          <ul className="list-disc list-inside text-sm text-gray-700">
            {sources.map((s, idx) => {
              const match = s.match(/^-\s*(.+?)\s*\(([^)]+)\)$/);
              if (match) {
                return (
                  <li key={idx}>
                    <a
                      href={match[2]}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline text-blue-700"
                    >
                      {match[1]}
                    </a>
                  </li>
                );
              }
              return <li key={idx}>{s.replace(/^-\s*/, "")}</li>;
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [allQuestions, setAllQuestions] = useState<string[]>([]);
  const [displayedQuestions, setDisplayedQuestions] = useState<string[]>([]);
  const [fade, setFade] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const hasUserMessage = messages.some((m) => m.role === "user");

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    async function loadPopularQuestions() {
      try {
        const res = await fetch("/popular_questions.json");
        if (!res.ok) throw new Error("failed to load popular questions");
        const data = await res.json();

        if (Array.isArray(data.questions)) {
          setAllQuestions(data.questions);
          setDisplayedQuestions(getRandomQuestions(data.questions, 4));
        } else {
          throw new Error("invalid JSON format");
        }
      } catch (err) {
        console.error("Error loading popular questions:", err);
        const fallback = [
          "What is the time limit for a master's degree?",
          "How many thesis credits must a master's student enroll in?",
          "Are Ph.D. students assigned a guidance committee?",
          "How many courses are required for a graduate certificate program?",
        ];
        setAllQuestions(fallback);
        setDisplayedQuestions(fallback);
      }
    }

    loadPopularQuestions();
  }, []);

  useEffect(() => {
    if (allQuestions.length <= 4) return;
    const interval = setInterval(() => {
      setFade(true);
      setTimeout(() => {
        setDisplayedQuestions(getRandomQuestions(allQuestions, 4));
        setFade(false);
      }, 300);
    }, 15000);
    return () => clearInterval(interval);
  }, [allQuestions]);

  function getRandomQuestions(arr: string[], n: number): string[] {
    const shuffled = [...arr].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, n);
  }

  const sendMessage = async (userInput: string) => {
    setMessages((prev) => [...prev, { role: "user", content: userInput }]);

    try {
      const res = await fetch(CHAT_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-Id": sessionId,
        },
        body: JSON.stringify({ message: userInput }),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      
      console.log("API Response:", data); // Debug log
      
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: data.answer,
          sources: data.sources,
          hasAlternative: data.has_alternative || false,
          alternativeAnswer: data.alternative_answer,
          alternativeSources: data.alternative_sources,
          answerMode: data.answer_mode,
          goldSimilarity: data.gold_similarity,
          selectedVersion: "primary"
        },
      ]);
    } catch (err) {
      console.error("API Error:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "Sorry, there was an error connecting to the chatbot API.",
          sources: [],
        },
      ]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userInput = input;
    setInput("");
    sendMessage(userInput);
  };

  const handleCardClick = (q: string) => {
    sendMessage(q);
  };

  const handleVersionSelect = (messageIndex: number, version: "primary" | "alternative") => {
    setMessages(prev => prev.map((msg, idx) => 
      idx === messageIndex ? { ...msg, selectedVersion: version } : msg
    ));
  };

  return (
    <main className="min-h-screen h-screen flex flex-col bg-[var(--unh-white)]">
      <header
        className="bg-[var(--unh-blue)] px-8 py-2 text-center shadow-md flex-none"
        style={{ color: "#fff", zIndex: 20, position: "relative" }}
      >
        <div className="flex items-center">
          <img
            src="/unh.svg"
            alt="UNH Logo"
            className="my-6 mr-4"
            style={{
              maxWidth: "125px",
              height: "auto",
              width: "auto",
              marginTop: "24px",
              marginBottom: "24px",
            }}
          />
          <span
            className="text-3xl font-bold"
            style={{ fontFamily: "Glypha, Arial, sans-serif" }}
          >
            GradCatBot
          </span>
        </div>
      </header>

      <div className="flex-1 flex flex-col items-center overflow-hidden min-h-0">
        {!hasUserMessage ? (
          <div className="flex-1 flex items-center justify-center w-full min-h-0 overflow-hidden">
            <section
              className="w-full max-w-3xl mx-auto px-6 py-4 flex flex-col flex-1 min-h-0 justify-center"
              style={{ zIndex: 0, position: "relative" }}
            >
              <h1 className="text-3xl font-bold mb-4 text-center text-gray-400 flex-none">
                What can I help with?
              </h1>
              <div
                className={`grid grid-cols-1 sm:grid-cols-2 gap-4 w-full transition-opacity duration-300 pr-3 pb-8 pt-1 pl-1 ${
                  fade ? "opacity-0" : "opacity-100"
                } max-h-full overflow-y-auto`}
                style={{ scrollbarGutter: 'stable both-edges' }}
              >
                {displayedQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleCardClick(q)}
                    className="bg-white shadow-lg rounded-xl p-3 sm:p-4 text-sm sm:text-base font-medium hover:bg-[var(--unh-light-gray)] hover:shadow-xl transition flex items-center justify-center text-center min-h-[88px]"
                    style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap', lineHeight: '1.4' }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </section>
          </div>
        ) : (
          <div className="w-2/3 flex flex-col h-full">
            <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-2">
              {messages.map((msg, i) => {
                const isRoleChange = i > 0 && messages[i - 1].role !== msg.role;
                
                if (msg.role === "user") {
                  return (
                    <div
                      key={i}
                      className={`flex justify-end items-end mb-2${
                        i === messages.length - 1 ? " mb-6" : ""
                      }`}
                      style={isRoleChange ? { marginTop: "1rem" } : {}}
                    >
                      <div className="bg-[var(--unh-blue)] text-[var(--unh-white)] rounded-2xl px-6 py-4 text-lg md:text-xl max-w-[800px] shadow-lg">
                        {msg.content}
                      </div>
                      <div className="flex-shrink-0 w-10 h-10 ml-2 mb-1">
                        <div className="w-10 h-10 bg-[var(--unh-blue)] rounded-full flex items-center justify-center">
                          <img src="/student.svg" alt="User" className="w-6 h-6" />
                        </div>
                      </div>
                    </div>
                  );
                } else {
                  const showDual = msg.hasAlternative && msg.alternativeAnswer;
                  const selectedVersion = msg.selectedVersion || "primary";
                  
                  return (
                    <div
                      key={i}
                      className={`flex justify-start items-start mb-2${
                        i === messages.length - 1 ? " mb-6" : ""
                      }`}
                      style={isRoleChange ? { marginTop: "1rem" } : {}}
                    >
                      <div className="flex-shrink-0 w-10 h-10 mr-2 mt-1">
                        <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center shadow-md">
                          <img src="/mascot.svg" alt="Bot" className="w-8 h-8" />
                        </div>
                      </div>
                      
                      <div className="max-w-[800px] w-full">
                        {showDual ? (
                          <div className="space-y-2">
                            <div className="text-sm text-gray-600 mb-2 px-2 font-medium">
                              Multiple answers available:
                            </div>
                            
                            <AnswerVersion
                              answer={msg.content}
                              sources={msg.sources}
                              isSelected={selectedVersion === "primary"}
                              onSelect={() => handleVersionSelect(i, "primary")}
                              label="Gold Standard Answer"
                              badge={msg.goldSimilarity ? `${(msg.goldSimilarity * 100).toFixed(0)}% match` : "Verified"}
                            />
                            
                            <AnswerVersion
                              answer={msg.alternativeAnswer!}
                              sources={msg.alternativeSources}
                              isSelected={selectedVersion === "alternative"}
                              onSelect={() => handleVersionSelect(i, "alternative")}
                              label="AI-Generated Answer"
                              badge="Retrieved"
                            />
                          </div>
                        ) : (
                          <div className="bg-[var(--unh-light-gray)] text-black rounded-2xl px-6 py-4 text-lg md:text-xl shadow-md">
                            <div dangerouslySetInnerHTML={{ __html: linkify(msg.content) }} />
                            
                            {msg.sources && msg.sources.length > 0 && (
                              <div className="mt-4">
                                <div className="font-semibold text-sm mb-1">Sources:</div>
                                <ul className="list-disc list-inside text-sm text-gray-700">
                                  {msg.sources.map((s, idx) => {
                                    const match = s.match(/^-\s*(.+?)\s*\(([^)]+)\)$/);
                                    if (match) {
                                      return (
                                        <li key={idx}>
                                          <a
                                            href={match[2]}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="underline text-blue-700"
                                          >
                                            {match[1]}
                                          </a>
                                        </li>
                                      );
                                    }
                                    return <li key={idx}>{s.replace(/^-\s*/, "")}</li>;
                                  })}
                                </ul>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                }
              })}
              <div ref={chatEndRef} />
            </div>
          </div>
        )}
      </div>

      <div className="w-full flex flex-col flex-none" style={{ background: 'white', zIndex: 10 }}>
        <div className="w-full px-4">
          <div className="mx-auto w-2/3 flex gap-2 py-2">
            <div className="flex-1 flex items-center">
              <div className="relative w-full flex items-center">
                <button
                  type="button"
                  onClick={async () => {
                    setInput("");
                    setMessages([]);
                    try {
                      await fetch("/reset", {
                        method: "POST",
                        headers: {
                          "Content-Type": "application/json",
                          "X-Session-Id": sessionId,
                        },
                      });
                    } catch (err) {
                      console.log("Reset signal failed:", err);
                    }
                  }}
                  className="absolute left-3 top-1/2 -translate-y-1/2 rounded-full p-2 flex items-center justify-center hover:bg-gray-100 transition-colors z-10"
                  aria-label="Clear input"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={2}
                    stroke="currentColor"
                    className="w-5 h-5 text-gray-600 hover:text-gray-800"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m-4.991 4.99a8.25 8.25 0 01-1.697 1.697"
                    />
                  </svg>
                </button>
                <input
                  className="w-full rounded-full border-2 border-gray-400 text-black pl-14 pr-24 py-6 text-lg md:text-xl placeholder:text-gray-400 bg-transparent box-border focus:outline-none"
                  type="text"
                  placeholder="Ask questions about programs, courses, and policies"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={handleSubmit}
                  className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-[var(--unh-blue)] p-3 flex items-center justify-center shadow hover:bg-[var(--unh-accent-blue)] transition-colors z-10"
                  aria-label="Send"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={2}
                    stroke="white"
                    className="w-6 h-6"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M5 12h14M12 5l7 7-7 7"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
        <div className="w-full px-4 pb-4 text-center text-sm text-gray-500">
          <p className="mx-auto w-2/3 text-lg">
            Not answering your question? Contact us at grad.school@unh.edu.
          </p>
        </div>
      </div>
    </main>
  );
}