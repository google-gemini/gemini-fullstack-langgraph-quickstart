import React, { ReactNode } from "react";
import type { Message } from "@langchain/langgraph-sdk";
import { motion } from "framer-motion";
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
    <h1 className={cn("text-2xl font-bold mt-4 mb-2", className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: MdComponentProps) => (
    <h2 className={cn("text-xl font-bold mt-3 mb-2", className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: MdComponentProps) => (
    <h3 className={cn("text-lg font-bold mt-3 mb-1", className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: MdComponentProps) => (
    <p className={cn("mb-3 leading-7", className)} {...props}>
      {children}
    </p>
  ),
  a: ({ className, children, href, ...props }: MdComponentProps) => (
    <Badge className="text-xs mx-0.5">
      <a
        className={cn("text-blue-400 hover:text-blue-300 text-xs", className)}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        {...props}
      >
        {children}
      </a>
    </Badge>
  ),
  ul: ({ className, children, ...props }: MdComponentProps) => (
    <ul className={cn("list-disc pl-6 mb-3", className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: MdComponentProps) => (
    <ol className={cn("list-decimal pl-6 mb-3", className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: MdComponentProps) => (
    <li className={cn("mb-1", className)} {...props}>
      {children}
    </li>
  ),
  blockquote: ({ className, children, ...props }: MdComponentProps) => (
    <blockquote
      className={cn(
        "border-l-4 border-neutral-600 pl-4 italic my-3 text-sm",
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
        "bg-neutral-900 rounded px-1 py-0.5 font-mono text-xs",
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
        "bg-neutral-900 p-3 rounded-lg overflow-x-auto font-mono text-xs my-3",
        className
      )}
      {...props}
    >
      {children}
    </pre>
  ),
  hr: ({ className, ...props }: MdComponentProps) => (
    <hr className={cn("border-neutral-600 my-4", className)} {...props} />
  ),
  table: ({ className, children, ...props }: MdComponentProps) => (
    <div className="my-3 overflow-x-auto">
      <table className={cn("border-collapse w-full", className)} {...props}>
        {children}
      </table>
    </div>
  ),
  th: ({ className, children, ...props }: MdComponentProps) => (
    <th
      className={cn(
        "border border-neutral-600 px-3 py-2 text-left font-bold",
        className
      )}
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ className, children, ...props }: MdComponentProps) => (
    <td
      className={cn("border border-neutral-600 px-3 py-2", className)}
      {...props}
    >
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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex items-start gap-3",
        isHuman ? "justify-end" : ""
      )}
    >
      <div
        className={cn(
          "relative break-words flex flex-col rounded-3xl min-h-7 max-w-[100%] sm:max-w-[90%] px-4 pt-3",
          isHuman
            ? "bg-neutral-700 text-white rounded-br-lg"
            : "bg-neutral-800 text-neutral-100 rounded-bl-lg"
        )}
      >
        {!isHuman && activityForThisBubble && activityForThisBubble.length > 0 && (
          <div className="mb-3 border-b border-neutral-700 pb-3 text-xs">
            <ActivityTimeline
              processedEvents={activityForThisBubble}
              isLoading={isLiveActivityForThisBubble}
            />
          </div>
        )}
        {renderContent()}
        {!isHuman && (
          <Button
            variant="default"
            className="cursor-pointer bg-neutral-700 border-neutral-600 text-neutral-300 self-end mt-2"
            onClick={() =>
              handleCopy(
                typeof message.content === "string"
                  ? message.content
                  : JSON.stringify(message.content, null, 2),
                message.id!
              )
            }
          >
            {copiedMessageId === message.id ? "Copied" : "Copy"}
            {copiedMessageId === message.id ? (
              <CopyCheck className="ml-2 h-4 w-4" />
            ) : (
              <Copy className="ml-2 h-4 w-4" />
            )}
          </Button>
        )}
      </div>
    </motion.div>
  );
};

export default React.memo(MessageBubble);
