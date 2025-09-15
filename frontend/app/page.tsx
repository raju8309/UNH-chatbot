"use client";


import React, { useState, useRef, useEffect } from "react";
import ArrowIcon from "./ArrowIcon";

export default function Home() {
  // Placeholder for chat messages
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [input, setInput] = useState("");

  // Ref for chat scroll
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Scroll to bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: input }]);
    setInput("");

    // Simulate bot response
    setTimeout(() => {
      setMessages(prev => [...prev, { role: "bot", content: "This is a placeholder response." }]);
    }, 1000);
  };

  return (
    <main className="h-screen flex flex-col bg-[var(--unh-white)] overflow-hidden">
      <header className="bg-[var(--unh-blue)] px-8 py-4 rounded-b-lg text-center shadow-md" style={{ color: '#fff' }}>
  <img src="/unh.svg" alt="UNH Logo" className="mx-auto my-6" style={{ maxWidth: '500px', height: 'auto', width: 'auto', marginTop: '24px', marginBottom: '24px' }} />
      </header>
      <div className="flex-1 flex flex-col items-center overflow-hidden">
        <div className="w-2/3 flex flex-col h-full">
          <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-2" style={{wordBreak: 'break-word'}}>
            {messages.map((msg, i) => (
              <div
                key={i}
                className={
                  msg.role === "user"
                    ? `flex justify-end mb-2${i === messages.length - 1 ? ' mb-6' : ''}`
                    : `flex justify-start mb-2${i === messages.length - 1 ? ' mb-6' : ''}`
                }
              >
                <span
                  className={
                    msg.role === "user"
                      ? "bg-[var(--unh-blue)] text-white px-8 py-4 rounded-full max-w-lg min-w-0 break-words shadow-md relative"
                      : "bg-gray-100 text-gray-900 px-8 py-4 rounded-full max-w-lg min-w-0 break-words shadow-md relative border border-gray-300"
                  }
                  style={{ position: 'relative', display: 'inline-block', wordBreak: 'break-word' }}
                >
                  {msg.content}
                  {msg.role === "user" ? (
                    <span
                      style={{
                        position: 'absolute',
                        right: -10,
                        bottom: 8,
                        width: 0,
                        height: 0,
                        borderTop: '8px solid transparent',
                        borderBottom: '8px solid transparent',
                        borderLeft: '8px solid #003366',
                        content: '""',
                      }}
                    />
                  ) : (
                    <span
                      style={{
                        position: 'absolute',
                        left: -10,
                        bottom: 8,
                        width: 0,
                        height: 0,
                        borderTop: '8px solid transparent',
                        borderBottom: '8px solid transparent',
                        borderRight: '8px solid #d1d5db',
                        content: '""',
                      }}
                    />
                  )}
                </span>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
          <form onSubmit={sendMessage} className="flex gap-2 bg-white py-2 px-4 border-t border-gray-200">
            <div className="flex-1">
              <input
                className="w-full border border-gray-300 rounded-full px-8 py-4 focus:outline-none focus:ring-2 focus:ring-[var(--unh-blue)]"
                type="text"
                placeholder="Ask questions about programs, courses, and policies"
                value={input}
                onChange={e => setInput(e.target.value)}
              />
            </div>
          </form>
          <div className="w-full px-4 pb-2 text-center text-sm text-gray-500">
            <p>Not answering your question? Contact us at grad.school@unh.edu.</p>
          </div>
        </div>
      </div>
    </main>
  );
}
