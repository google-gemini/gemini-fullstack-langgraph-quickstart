import { Loader2 } from "lucide-react";

export function LoadingBubble() {
  return (
    <div className="flex items-center justify-center p-4">
      <Loader2 className="h-8 w-8 animate-spin text-neutral-400" />
    </div>
  );
}
