import { useState } from "react";
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
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Globe, Server, Settings2, Network } from "lucide-react"; // Added Network icon

// Updated InputFormProps
interface InputFormProps {
  onSubmit: (inputValue: string, effort: string, model: string, provider: string, langsmithEnabled: boolean, searchMode: string) => void; // Added searchMode
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
  const [model, setModel] = useState("gemini-1.5-pro");
  const [selectedProvider, setSelectedProvider] = useState("gemini");
  const [langsmithEnabled, setLangsmithEnabled] = useState(true);
  const [searchMode, setSearchMode] = useState("internet_only"); // New state for Search Scope

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue, effort, model, selectedProvider, langsmithEnabled, searchMode); // Pass searchMode
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

  return (
    <form
      onSubmit={handleInternalSubmit}
      className={`flex flex-col gap-2 p-3`} // Removed rounded-b-xl from here, will be on parent if needed
    >
      {/* Main input area */}
      <div
        className={`flex flex-row items-center justify-between text-foreground rounded-xl ${ // text-foreground for light theme
          hasHistory ? "rounded-br-sm" : "" // This logic might need review based on overall page structure
        } break-words min-h-7 bg-card border border-border px-4 pt-3 shadow-sm`} // bg-card, border-border for light theme
      >
        <Textarea
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleInternalKeyDown}
          placeholder="Ask anything... e.g., Who won the Euro 2024 and scored the most goals?"
          className={`w-full text-foreground placeholder-muted-foreground resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none
                        md:text-base min-h-[56px] max-h-[200px] bg-transparent`} // bg-transparent, updated text colors
          rows={1}
        />
        <div className="-mt-3">
          {isLoading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-destructive hover:text-destructive/90 hover:bg-destructive/10 p-2 cursor-pointer rounded-full transition-all duration-200"
              onClick={onCancel}
            >
              <StopCircle className="h-5 w-5" />
            </Button>
          ) : (
            <Button
              type="submit"
              variant="ghost" // Using ghost, color comes from text-primary
              className={`${
                isSubmitDisabled
                  ? "text-muted-foreground" // Muted for disabled
                  : "text-primary hover:text-primary/90 hover:bg-primary/10" // Primary color for active
              } p-2 cursor-pointer rounded-full transition-all duration-200 text-base`}
              disabled={isSubmitDisabled}
            >
              Search
              <Send className="h-5 w-5" />
            </Button>
          )}
        </div>
      </div>

      {/* Configuration options row */}
      <div className="flex items-center justify-between flex-wrap gap-2"> {/* Added flex-wrap and gap-2 for better responsiveness */}
        <div className="flex flex-row gap-2 flex-wrap"> {/* Added flex-wrap here too */}
          {/* Effort Dropdown */}
          <div className="flex flex-row items-center gap-1 bg-card border border-border text-foreground focus-within:ring-1 focus-within:ring-ring rounded-lg pl-2 shadow-sm">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <Label htmlFor="effort-select" className="text-sm text-muted-foreground pr-1">Effort</Label>
            <Select value={effort} onValueChange={setEffort}>
              <SelectTrigger id="effort-select" className="w-[110px] bg-transparent border-none focus:ring-0">
                <SelectValue placeholder="Effort" />
              </SelectTrigger>
              <SelectContent className="bg-popover text-popover-foreground border-border">
                <SelectItem value="low" className="focus:bg-accent focus:text-accent-foreground">Low</SelectItem>
                <SelectItem value="medium" className="focus:bg-accent focus:text-accent-foreground">Medium</SelectItem>
                <SelectItem value="high" className="focus:bg-accent focus:text-accent-foreground">High</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Reasoning Model Dropdown */}
          <div className="flex flex-row items-center gap-1 bg-card border border-border text-foreground focus-within:ring-1 focus-within:ring-ring rounded-lg pl-2 shadow-sm">
            <Cpu className="h-4 w-4 text-muted-foreground" />
            <Label htmlFor="model-select" className="text-sm text-muted-foreground pr-1">Model</Label>
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger id="model-select" className="w-[150px] bg-transparent border-none focus:ring-0">
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent className="bg-popover text-popover-foreground border-border">
                <SelectItem value="gemini-1.5-flash" className="focus:bg-accent focus:text-accent-foreground">
                  <div className="flex items-center"><Zap className="h-4 w-4 mr-2 text-yellow-500" />Flash/Fast</div>
                </SelectItem>
                <SelectItem value="gemini-1.5-pro" className="focus:bg-accent focus:text-accent-foreground">
                  <div className="flex items-center"><Cpu className="h-4 w-4 mr-2 text-blue-500" />Pro/Advanced</div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* LLM Provider Dropdown */}
          <div className="flex flex-row items-center gap-1 bg-card border border-border text-foreground focus-within:ring-1 focus-within:ring-ring rounded-lg pl-2 shadow-sm">
            <Server className="h-4 w-4 text-muted-foreground" />
            <Label htmlFor="provider-select" className="text-sm text-muted-foreground pr-1">Provider</Label>
            <Select value={selectedProvider} onValueChange={setSelectedProvider}>
              <SelectTrigger id="provider-select" className="w-[140px] bg-transparent border-none focus:ring-0">
                <SelectValue placeholder="Provider" />
              </SelectTrigger>
              <SelectContent className="bg-popover text-popover-foreground border-border">
                <SelectItem value="gemini" className="focus:bg-accent focus:text-accent-foreground">
                  <div className="flex items-center"><Globe className="h-4 w-4 mr-2 text-blue-500" />Gemini</div>
                </SelectItem>
                <SelectItem value="openrouter" className="focus:bg-accent focus:text-accent-foreground">
                  <div className="flex items-center"><Server className="h-4 w-4 mr-2 text-green-500" />OpenRouter</div>
                </SelectItem>
                <SelectItem value="deepseek" className="focus:bg-accent focus:text-accent-foreground">
                  <div className="flex items-center"><Server className="h-4 w-4 mr-2 text-teal-500" />DeepSeek</div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* LangSmith Toggle */}
          <div className="flex items-center space-x-2 bg-card border border-border text-foreground rounded-lg px-3 py-[7px] shadow-sm"> {/* Adjusted padding to match selects */}
            <Settings2 className="h-4 w-4 text-muted-foreground" />
            <Switch
              id="langsmith-toggle"
              checked={langsmithEnabled}
              onCheckedChange={setLangsmithEnabled}
              // className="data-[state=checked]:bg-primary data-[state=unchecked]:bg-input" // Using default Switch colors which should adapt
            />
            <Label htmlFor="langsmith-toggle" className="text-sm text-muted-foreground cursor-pointer">
              LangSmith
            </Label>
          </div>

          {/* Search Scope Dropdown */}
          <div className="flex flex-row items-center gap-1 bg-card border border-border text-foreground focus-within:ring-1 focus-within:ring-ring rounded-lg pl-2 shadow-sm">
            <Network className="h-4 w-4 text-muted-foreground" />
            <Label htmlFor="scope-select" className="text-sm text-muted-foreground pr-1">Scope</Label>
            <Select value={searchMode} onValueChange={setSearchMode}>
              <SelectTrigger id="scope-select" className="w-[180px] bg-transparent border-none focus:ring-0">
                <SelectValue placeholder="Search Scope" />
              </SelectTrigger>
              <SelectContent className="bg-popover text-popover-foreground border-border">
                <SelectItem value="internet_only" className="focus:bg-accent focus:text-accent-foreground">Internet Only</SelectItem>
                <SelectItem value="local_only" className="focus:bg-accent focus:text-accent-foreground">Local Network Only</SelectItem>
                <SelectItem value="internet_then_local" className="focus:bg-accent focus:text-accent-foreground">Internet, then Local</SelectItem>
                <SelectItem value="local_then_internet" className="focus:bg-accent focus:text-accent-foreground">Local, then Internet</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* New Search Button */}
        {hasHistory && (
          <Button
            className="bg-card hover:bg-accent text-accent-foreground border border-border shadow-sm" // Updated styles
            variant="outline" // Using outline variant for a less prominent look
            onClick={() => window.location.reload()}
          >
            <SquarePen size={16} className="mr-2" /> {/* Added margin to icon */}
            New Search
          </Button>
        )}
      </div>
    </form>
  );
};
