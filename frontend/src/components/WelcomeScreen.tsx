import { InputForm } from "./InputForm";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string,
    model: string,
    provider: string,
    langsmithEnabled: boolean,
    searchMode: string // Added searchMode
  ) => void;
  onCancel: () => void;
  isLoading: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
}) => (
  <div className="flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-4">
    <div>
      {/* Updated text colors for light theme */}
      <h1 className="text-5xl md:text-6xl font-semibold text-foreground mb-3">
        Welcome.
      </h1>
      <p className="text-xl md:text-2xl text-muted-foreground">
        How can I help you today?
      </p>
    </div>
    <div className="w-full mt-4">
      <InputForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={false}
      />
    </div>
    {/* Updated text color for "Powered by" text */}
    <p className="text-xs text-muted-foreground/80">
      Powered by Google Gemini and LangChain LangGraph.
    </p>
  </div>
);
