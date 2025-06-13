import type React from "react";
import type { Message } from "@langchain/langgraph-sdk";
import { ScrollArea } from "@/components/ui/scroll-area";
import { InputForm } from "@/components/InputForm";
import { useState } from "react";
import type { ProcessedEvent } from "@/components/ActivityTimeline";
import MessageBubble from "./chat/MessageBubble";
import { LoadingBubble } from "./chat/LoadingBubble";

interface ChatMessagesViewProps {
  messages: Message[];
  isLoading: boolean;
  scrollAreaRef: React.RefObject<HTMLDivElement | null>;
  onSubmit: (inputValue: string, effort: string, model: string) => void;
  onCancel: () => void;
  liveActivityEvents: ProcessedEvent[];
  historicalActivities: Record<string, ProcessedEvent[]>;
}

export function ChatMessagesView({
  messages,
  isLoading,
  scrollAreaRef,
  onSubmit,
  onCancel,
  liveActivityEvents,
  historicalActivities,
}: ChatMessagesViewProps) {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  const handleCopy = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-grow" ref={scrollAreaRef}>
        <div className="p-4 md:p-6 space-y-4 max-w-4xl mx-auto pt-16">
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id || `msg-${index}`}
              message={message}
              isLastMessage={index === messages.length - 1}
              isOverallLoading={isLoading}
              historicalActivity={historicalActivities[message.id!]}
              liveActivity={liveActivityEvents}
              handleCopy={handleCopy}
              copiedMessageId={copiedMessageId}
            />
          ))}
          {isLoading &&
            (messages.length === 0 ||
              messages[messages.length - 1].type === "human") && <LoadingBubble />}
        </div>
      </ScrollArea>
      <InputForm
        onSubmit={onSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={messages.length > 0}
      />
    </div>
  );
}
