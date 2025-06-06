import type React from "react";
import type { Message } from "@langchain/langgraph-sdk";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Copy, CopyCheck } from "lucide-react";
import { InputForm } from "@/components/InputForm";
import { Button } from "@/components/ui/button";
import { useState, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { cn } from "@/lib/utils";
// Badge removed as link styling is now direct
import {
  ActivityTimeline,
  ProcessedEvent,
} from "@/components/ActivityTimeline";

// Markdown component props type
type MdComponentProps = {
  className?: string;
  children?: ReactNode;
  [key: string]: any;
};

// Updated Markdown components for light theme
const mdComponents = {
  h1: ({ className, children, ...props }: MdComponentProps) => (
    <h1 className={cn("text-2xl font-bold mt-4 mb-2 text-foreground", className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: MdComponentProps) => (
    <h2 className={cn("text-xl font-bold mt-3 mb-2 text-foreground", className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: MdComponentProps) => (
    <h3 className={cn("text-lg font-bold mt-3 mb-1 text-foreground", className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: MdComponentProps) => (
    <p className={cn("mb-3 leading-7 text-foreground", className)} {...props}>
      {children}
    </p>
  ),
  a: ({ className, children, href, ...props }: MdComponentProps) => (
    <a
      className={cn("text-primary hover:text-primary/80 underline", className)}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    >
      {children}
    </a>
  ),
  ul: ({ className, children, ...props }: MdComponentProps) => (
    <ul className={cn("list-disc pl-6 mb-3 text-foreground", className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: MdComponentProps) => (
    <ol className={cn("list-decimal pl-6 mb-3 text-foreground", className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: MdComponentProps) => (
    <li className={cn("mb-1 text-foreground", className)} {...props}>
      {children}
    </li>
  ),
  blockquote: ({ className, children, ...props }: MdComponentProps) => (
    <blockquote
      className={cn(
        "border-l-4 border-border pl-4 italic my-3 text-muted-foreground",
        className
      )}
      {...props}
    >
      {children}
    </blockquote>
  ),
  code: ({ className, children, ...props }: MdComponentProps) => (
    <code
      className={cn(
        "bg-muted text-muted-foreground rounded px-1 py-0.5 font-mono text-xs",
        className
      )}
      {...props}
    >
      {children}
    </code>
  ),
  pre: ({ className, children, ...props }: MdComponentProps) => (
    <pre
      className={cn(
        "bg-muted text-muted-foreground p-3 rounded-lg overflow-x-auto font-mono text-xs my-3",
        className
      )}
      {...props}
    >
      {children}
    </pre>
  ),
  hr: ({ className, ...props }: MdComponentProps) => (
    <hr className={cn("border-border my-4", className)} {...props} />
  ),
  table: ({ className, children, ...props }: MdComponentProps) => (
    <div className="my-3 overflow-x-auto">
      <table className={cn("border-collapse w-full border border-border", className)} {...props}>
        {children}
      </table>
    </div>
  ),
  th: ({ className, children, ...props }: MdComponentProps) => (
    <th
      className={cn(
        "border border-border px-3 py-2 text-left font-bold text-foreground", // Changed from card-foreground
        className
      )}
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ className, children, ...props }: MdComponentProps) => (
    <td
      className={cn("border border-border px-3 py-2 text-foreground", className)} // Changed from card-foreground
      {...props}
    >
      {children}
    </td>
  ),
};

// Props for HumanMessageBubble
interface HumanMessageBubbleProps {
  message: Message;
  mdComponents: typeof mdComponents;
}

// HumanMessageBubble Component - Updated for light theme
const HumanMessageBubble: React.FC<HumanMessageBubbleProps> = ({
  message,
  mdComponents,
}) => {
  return (
    <div
      className={`text-primary-foreground rounded-xl break-words min-h-7 bg-primary max-w-[100%] sm:max-w-[90%] px-4 py-3 shadow rounded-br-lg`}
    >
      <ReactMarkdown components={mdComponents}>
        {typeof message.content === "string"
          ? message.content
          : JSON.stringify(message.content)}
      </ReactMarkdown>
    </div>
  );
};

// Props for AiMessageBubble
interface AiMessageBubbleProps {
  message: Message;
  historicalActivity: ProcessedEvent[] | undefined;
  liveActivity: ProcessedEvent[] | undefined;
  isLastMessage: boolean;
  isOverallLoading: boolean;
  mdComponents: typeof mdComponents;
  handleCopy: (text: string, messageId: string) => void;
  copiedMessageId: string | null;
}

// AiMessageBubble Component - Updated for light theme
const AiMessageBubble: React.FC<AiMessageBubbleProps> = ({
  message,
  historicalActivity,
  liveActivity,
  isLastMessage,
  isOverallLoading,
  mdComponents,
  handleCopy,
  copiedMessageId,
}) => {
  const activityForThisBubble =
    isLastMessage && isOverallLoading ? liveActivity : historicalActivity;
  const isLiveActivityForThisBubble = isLastMessage && isOverallLoading;

  return (
    <div className={`relative break-words flex flex-col bg-card text-card-foreground p-4 rounded-xl shadow rounded-bl-none`}>
      {activityForThisBubble && activityForThisBubble.length > 0 && (
        <div className="mb-3 border-b border-border pb-3 text-xs text-muted-foreground">
          <ActivityTimeline
            processedEvents={activityForThisBubble}
            isLoading={isLiveActivityForThisBubble}
          />
        </div>
      )}
      <ReactMarkdown components={mdComponents}>
        {typeof message.content === "string"
          ? message.content
          : JSON.stringify(message.content)}
      </ReactMarkdown>
      <Button
        variant="outline"
        size="sm"
        className="cursor-pointer bg-card hover:bg-accent text-muted-foreground hover:text-accent-foreground border-border self-end mt-2"
        onClick={() =>
          handleCopy(
            typeof message.content === "string"
              ? message.content
              : JSON.stringify(message.content),
            message.id!
          )
        }
      >
        {copiedMessageId === message.id ? <CopyCheck className="h-3 w-3 mr-1" /> : <Copy className="h-3 w-3 mr-1" />}
        {copiedMessageId === message.id ? "Copied" : "Copy"}
      </Button>
    </div>
  );
};

interface ChatMessagesViewProps {
  messages: Message[];
  isLoading: boolean;
  scrollAreaRef: React.RefObject<HTMLDivElement | null>;
  // Ensure this onSubmit matches the one in App.tsx after all modifications
  onSubmit: (inputValue: string, effort: string, model: string, provider: string, langsmithEnabled: boolean, searchMode: string) => void;
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
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-grow" ref={scrollAreaRef}>
        <div className="p-4 md:p-6 space-y-4 max-w-4xl mx-auto pt-16"> {/* Increased space-y for more separation */}
          {messages.map((message, index) => {
            const isLast = index === messages.length - 1;
            return (
              <div key={message.id || `msg-${index}`} className="space-y-3">
                <div
                  className={`flex items-start gap-3 ${
                    message.type === "human" ? "justify-end" : ""
                  }`}
                >
                  {message.type === "human" ? (
                    <HumanMessageBubble
                      message={message}
                      mdComponents={mdComponents}
                    />
                  ) : (
                    <div className="w-full max-w-[100%] sm:max-w-[90%]"> {/* Wrapper for AI bubble to control width */}
                      <AiMessageBubble
                        message={message}
                        historicalActivity={historicalActivities[message.id!]}
                        liveActivity={liveActivityEvents}
                        isLastMessage={isLast}
                        isOverallLoading={isLoading}
                        mdComponents={mdComponents}
                        handleCopy={handleCopy}
                        copiedMessageId={copiedMessageId}
                      />
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          {isLoading &&
            (messages.length === 0 ||
              messages[messages.length - 1].type === "human") && (
              <div className="flex items-start gap-3 mt-3">
                <div className="w-full max-w-[100%] sm:max-w-[90%]"> {/* Wrapper for AI loading bubble */}
                  <div className="relative group rounded-xl p-4 shadow break-words bg-card text-muted-foreground rounded-bl-none min-h-[56px]"> {/* Updated styles */}
                    {liveActivityEvents.length > 0 ? (
                      <div className="text-xs">
                        <ActivityTimeline
                          processedEvents={liveActivityEvents}
                          isLoading={true}
                        />
                      </div>
                    ) : (
                      <div className="flex items-center justify-start h-full">
                        <Loader2 className="h-5 w-5 animate-spin mr-2" />
                        <span>Processing...</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
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
