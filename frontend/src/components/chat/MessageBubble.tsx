import React, { ReactNode } from "react";
import type { Message } from "@langchain/langgraph-sdk";
import { motion, AnimatePresence } from "framer-motion"; // Added AnimatePresence
import ReactMarkdown from "react-markdown";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import JsonView from "./JsonView";
import {
  ActivityTimeline,
  ProcessedEvent,
} from "@/components/ActivityTimeline";
import { Button } from "@/components/ui/button";
import { Copy, CopyCheck } from "lucide-react";

// Markdown component props type
type MdComponentProps = {
  className?: string;
  children?: ReactNode;
  href?: string;
  [key: string]: any;
};

// Markdown components
const mdComponents: any = {
  h1: ({ className, children, ...props }: MdComponentProps) => (
    <h1
      className={cn("text-2xl font-semibold mt-6 mb-3", className)}
      {...props}
    >
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: MdComponentProps) => (
    <h2
      className={cn("text-xl font-semibold mt-5 mb-2", className)}
      {...props}
    >
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: MdComponentProps) => (
    <h3
      className={cn("text-lg font-semibold mt-4 mb-2", className)}
      {...props}
    >
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: MdComponentProps) => (
    <p className={cn("mb-4 leading-relaxed", className)} {...props}>
      {children}
    </p>
  ),
  a: ({ className, children, href, ...props }: MdComponentProps) => (
    <a
      className={cn(
        "text-primary hover:text-primary/80 underline",
        className
      )}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    >
      {children}
    </a>
  ),
  ul: ({ className, children, ...props }: MdComponentProps) => (
    <ul className={cn("list-disc pl-5 mb-4 space-y-1", className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: MdComponentProps) => (
    <ol
      className={cn("list-decimal pl-5 mb-4 space-y-1", className)}
      {...props}
    >
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: MdComponentProps) => (
    <li className={cn("", className)} {...props}>
      {children}
    </li>
  ),
  blockquote: ({ className, children, ...props }: MdComponentProps) => (
    <blockquote
      className={cn(
        "border-l-4 border-border pl-4 italic my-4 text-muted-foreground",
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
        "bg-muted text-muted-foreground rounded px-1.5 py-1 font-mono text-sm",
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
        "bg-muted text-muted-foreground p-4 rounded-lg overflow-x-auto font-mono text-sm my-4",
        className
      )}
      {...props}
    >
      {children}
    </pre>
  ),
  hr: ({ className, ...props }: MdComponentProps) => (
    <hr className={cn("border-border my-6", className)} {...props} />
  ),
  table: ({ className, children, ...props }: MdComponentProps) => (
    <div className="my-4 overflow-x-auto rounded-lg border border-border">
      <table
        className={cn("w-full divide-y divide-border", className)}
        {...props}
      >
        {children}
      </table>
    </div>
  ),
  th: ({ className, children, ...props }: MdComponentProps) => (
    <th
      className={cn(
        "px-4 py-3 text-left font-medium bg-muted text-muted-foreground",
        className
      )}
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ className, children, ...props }: MdComponentProps) => (
    <td className={cn("px-4 py-3 text-sm", className)} {...props}>
      {children}
    </td>
  ),
};

interface MessageBubbleProps {
  message: Message;
  historicalActivity?: ProcessedEvent[];
  liveActivity?: ProcessedEvent[];
  isLastMessage: boolean;
  isOverallLoading: boolean;
  handleCopy: (text: string, messageId: string) => void;
  copiedMessageId: string | null;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  historicalActivity,
  liveActivity,
  isLastMessage,
  isOverallLoading,
  handleCopy,
  copiedMessageId,
}) => {
  const isHuman = message.type === "human";

  const renderContent = () => {
    if (typeof message.content === "string") {
      return <ReactMarkdown components={mdComponents}>{message.content}</ReactMarkdown>;
    }
    if (typeof message.content === 'object' && message.content !== null) {
        try {
            return <JsonView data={message.content as object} />;
        } catch {
            return <ReactMarkdown components={mdComponents}>{JSON.stringify(message.content)}</ReactMarkdown>;
        }
    }
    return <ReactMarkdown components={mdComponents}>{JSON.stringify(message.content)}</ReactMarkdown>;
  };

  const activityForThisBubble =
    isLastMessage && isOverallLoading ? liveActivity : historicalActivity;
  const isLiveActivityForThisBubble = isLastMessage && isOverallLoading;

  return (
    <motion.div
      layout // Added layout prop to the main bubble container
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={cn("flex items-start gap-3", isHuman ? "justify-end" : "")}
    >
      <motion.div // Also add layout to the inner container that holds all content
        layout
        className={cn(
          "relative break-words flex flex-col rounded-xl min-h-7 max-w-[100%] sm:max-w-[90%] p-4 shadow-sm",
          isHuman
            ? "bg-primary text-primary-foreground rounded-br-md"
            : "bg-card text-card-foreground rounded-bl-md"
        )}
      >
        <AnimatePresence> {/* Handles appearance/disappearance of timeline */}
          {!isHuman && activityForThisBubble && activityForThisBubble.length > 0 && (
            <motion.div
              layout // Ensures the div itself animates layout changes (e.g. when timeline inside expands)
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2, ease: "easeInOut" }}
              className="mb-3 border-b border-border pb-3 text-xs overflow-hidden" // overflow-hidden helps with height animation
            >
              <ActivityTimeline
                processedEvents={activityForThisBubble}
                isLoading={isLiveActivityForThisBubble}
              />
            </motion.div>
          )}
        </AnimatePresence>
        <div className="prose prose-sm dark:prose-invert max-w-none prose-p:leading-normal prose-headings:font-medium prose-headings:text-foreground">
          {renderContent()}
        </div>
        {!isHuman && (
          <Button
            variant="ghost" // More subtle button variant
            size="icon"
            className="text-muted-foreground hover:text-foreground hover:bg-muted cursor-pointer self-end mt-3 transition-all duration-200 ease-in-out"
            onClick={() =>
              handleCopy(
                typeof message.content === "string"
                  ? message.content
                  : JSON.stringify(message.content, null, 2),
                message.id!
              )
            }
            aria-label={copiedMessageId === message.id ? "Message copied" : "Copy message content"} // Updated title to aria-label
          >
            {copiedMessageId === message.id ? (
              <CopyCheck className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </Button>
        )}
      </div>
    </motion.div>
  );
};

export default React.memo(MessageBubble);
