import { useState } from "react";
import { motion } from "framer-motion"; // Import motion
import { Button } from "@/components/ui/button";
import { SquarePen, Brain, Send, StopCircle, Zap, Cpu } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Updated InputFormProps
interface InputFormProps {
  onSubmit: (inputValue: string, effort: string, model: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
}

export const InputForm: React.FC<InputFormProps> = ({
  onSubmit,
  onCancel,
  isLoading,
  hasHistory,
}) => {
  const [internalInputValue, setInternalInputValue] = useState("");
  const [effort, setEffort] = useState("medium");
  const [model, setModel] = useState("gemini-2.5-flash-preview-04-17");

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue, effort, model);
    setInternalInputValue("");
  };

  const handleInternalKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const isSubmitDisabled = !internalInputValue.trim() || isLoading;
  const MotionButton = motion(Button); // Create motion component from Button

  return (
    <form
      onSubmit={handleInternalSubmit}
      className="flex flex-col gap-3 p-4"
    >
      <div
        className={`flex flex-row items-end gap-2 bg-card text-foreground p-3 rounded-lg break-words min-h-[56px]`}
      >
        <Textarea
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleInternalKeyDown}
          placeholder="Who won the Euro 2024 and scored the most goals?"
          className="w-full bg-transparent text-foreground placeholder-muted-foreground resize-none border-0 focus-visible:ring-1 focus-visible:ring-ring shadow-none md:text-base min-h-[56px] max-h-[200px] p-2 transition-shadow duration-200 ease-in-out focus:shadow-md"
          rows={1}
        />
        <div className="flex-shrink-0">
          {isLoading ? (
            <MotionButton
              type="button"
              variant="ghost"
              size="icon"
              className="text-destructive hover:text-destructive/90 hover:bg-destructive/10 p-2 rounded-full transition-colors duration-200 ease-in-out"
              onClick={onCancel}
              whileTap={{ scale: 0.90, opacity: 0.8 }}
              aria-label="Cancel generation" // Added aria-label
            >
              <StopCircle className="h-5 w-5" />
            </MotionButton>
          ) : (
            <MotionButton
              type="submit"
              variant="ghost"
              size="icon"
              className={`${
                isSubmitDisabled
                  ? "text-muted-foreground cursor-not-allowed"
                  : "text-primary hover:text-primary/90 hover:bg-primary/10"
              } p-2 rounded-full transition-colors duration-200 ease-in-out`}
              disabled={isSubmitDisabled}
              whileTap={{ scale: 0.90, opacity: 0.8 }}
              aria-label="Send message" // Added aria-label
            >
              <Send className="h-5 w-5" />
            </MotionButton>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-row gap-2">
          {/* Effort Select */}
          <div className="flex flex-row items-center gap-2 bg-card border border-border text-foreground focus-within:ring-1 focus-within:ring-ring rounded-lg pl-3 pr-1 py-1 transition-all duration-200 ease-in-out">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span id="effort-label" className="text-sm text-muted-foreground">Effort</span> {/* Added id for label */}
            <Select value={effort} onValueChange={setEffort}>
              <SelectTrigger
                aria-label="Select Effort" /* Direct aria-label */
                className="w-auto bg-transparent border-0 text-foreground focus:ring-0 focus:ring-offset-0 p-1 h-auto"
              >
                <SelectValue placeholder="Effort" />
              </SelectTrigger>
              <SelectContent className="bg-popover border-border text-popover-foreground cursor-pointer">
                <SelectItem
                  value="low"
                  className="hover:bg-muted focus:bg-muted cursor-pointer transition-colors duration-150 ease-in-out"
                >
                  Low
                </SelectItem>
                <SelectItem
                  value="medium"
                  className="hover:bg-muted focus:bg-muted cursor-pointer transition-colors duration-150 ease-in-out"
                >
                  Medium
                </SelectItem>
                <SelectItem
                  value="high"
                  className="hover:bg-muted focus:bg-muted cursor-pointer transition-colors duration-150 ease-in-out"
                >
                  High
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          {/* Model Select */}
          <div className="flex flex-row items-center gap-2 bg-card border border-border text-foreground focus-within:ring-1 focus-within:ring-ring rounded-lg pl-3 pr-1 py-1 transition-all duration-200 ease-in-out">
            <Cpu className="h-4 w-4 text-muted-foreground" />
            <span id="model-label" className="text-sm text-muted-foreground">Model</span> {/* Added id for label */}
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger
                aria-label="Select Model" /* Direct aria-label */
                className="w-auto bg-transparent border-0 text-foreground focus:ring-0 focus:ring-offset-0 p-1 h-auto"
              >
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent className="bg-popover border-border text-popover-foreground cursor-pointer">
                <SelectItem
                  value="gemini-2.0-flash"
                  className="hover:bg-muted focus:bg-muted cursor-pointer transition-colors duration-150 ease-in-out"
                >
                  <div className="flex items-center">
                    <Zap className="h-4 w-4 mr-2 text-yellow-400" /> 2.0 Flash
                  </div>
                </SelectItem>
                <SelectItem
                  value="gemini-2.5-flash-preview-04-17"
                  className="hover:bg-muted focus:bg-muted cursor-pointer transition-colors duration-150 ease-in-out"
                >
                  <div className="flex items-center">
                    <Zap className="h-4 w-4 mr-2 text-orange-400" /> 2.5 Flash
                  </div>
                </SelectItem>
                <SelectItem
                  value="gemini-2.5-pro-preview-05-06"
                  className="hover:bg-muted focus:bg-muted cursor-pointer transition-colors duration-150 ease-in-out"
                >
                  <div className="flex items-center">
                    <Cpu className="h-4 w-4 mr-2 text-purple-400" /> 2.5 Pro
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        {hasHistory && (
          <MotionButton
            className="bg-card hover:bg-muted border border-border text-foreground cursor-pointer rounded-lg px-3 py-2 text-sm transition-colors duration-200 ease-in-out"
            variant="default"
            onClick={() => window.location.reload()}
            whileTap={{ scale: 0.95, opacity: 0.9 }}
          >
            <SquarePen size={16} className="mr-2" />
            New Search
          </MotionButton>
        )}
      </div>
    </form>
  );
};
