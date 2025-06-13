import { InputForm } from "./InputForm";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string,
    model: string
  ) => void;
  onCancel: () => void;
  isLoading: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
}) => (
  <div className="flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-6 md:gap-8"> {/* Increased gap */}
    <div className="space-y-2"> {/* Added space-y for better control if more elements are added */}
      <h1 className="text-5xl md:text-6xl font-semibold text-foreground mb-2 md:mb-3"> {/* Updated text color, adjusted margin */}
        Welcome.
      </h1>
      <p className="text-xl md:text-2xl text-muted-foreground"> {/* Updated text color */}
        How can I help you today?
      </p>
    </div>
    <div className="w-full mt-6 md:mt-8"> {/* Increased top margin */}
      <InputForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={false} // Welcome screen implies no prior history for this session view
      />
    </div>
    <p className="text-xs text-muted-foreground/70"> {/* Updated text color with alpha for subtlety */}
      Powered by Google Gemini and LangChain LangGraph.
    </p>
  </div>
);
