import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2,
  Info,
  ChevronDown,
  // ChevronUp, // Will use motion.div to rotate ChevronDown
} from "lucide-react";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from 'framer-motion'; // Import motion
import TimelineEvent from './chat/TimelineEvent';

export interface ProcessedEvent {
  title: string;
  data: any;
}

interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  isLoading: boolean;
}

export function ActivityTimeline({
  processedEvents,
  isLoading,
}: ActivityTimelineProps) {
  const [isTimelineCollapsed, setIsTimelineCollapsed] =
    useState<boolean>(false);

  useEffect(() => {
    if (!isLoading && processedEvents.length > 0) {
      setIsTimelineCollapsed(true);
    }
  }, [isLoading, processedEvents]);

  return (
    <Card className="border-none rounded-lg bg-muted/30 dark:bg-muted/10 max-h-96 shadow-none">
      <CardHeader className="p-3"> {/* Reduced padding for header */}
        <CardDescription className="flex items-center justify-between text-xs">
          <div
            role="button" // Added role
            tabIndex={0} // Make focusable
            aria-expanded={!isTimelineCollapsed} // ARIA expanded state
            // aria-controls="timeline-content-area" // If the motion.div below had id="timeline-content-area"
            className="flex items-center justify-start w-full cursor-pointer gap-1 text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring rounded-sm" // Added focus styling
            onClick={() => setIsTimelineCollapsed(!isTimelineCollapsed)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsTimelineCollapsed(!isTimelineCollapsed); }} // Keyboard activation
          >
            <motion.div
              animate={{ rotate: isTimelineCollapsed ? 0 : 180 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronDown className="h-4 w-4" /> {/* Adjusted margin by removing mr-1, added to text instead */}
            </motion.div>
            <span className="ml-1">Research Activity</span> {/* Added ml-1 to text for spacing */}
          </div>
        </CardDescription>
      </CardHeader>
      <AnimatePresence initial={false}> {/* initial={false} to prevent animation on mount */}
        {!isTimelineCollapsed && (
          <motion.div
            key="timeline-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden" // Crucial for height animation
          >
            <ScrollArea className="max-h-80 overflow-y-auto">
              <CardContent className="p-3 pt-0">
                <AnimatePresence> {/* This AnimatePresence is for the TimelineEvents inside */}
                  {processedEvents.map((event, index) => (
                <TimelineEvent
                  key={index}
                  event={event}
                  isLast={index === processedEvents.length - 1}
                  isLoading={isLoading}
                />
              ))}
            </AnimatePresence>
            {isLoading && processedEvents.length === 0 && (
              <div className="flex items-center justify-center p-4 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin mr-2" />
                <p className="text-sm">Searching...</p>
              </div>
            )}
            {!isLoading && processedEvents.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground pt-8 pb-4">
                <Info className="h-5 w-5 mb-2" />
                <p className="text-sm">No activity to display.</p>
                <p className="text-xs mt-1">
                  Timeline will update during processing.
                </p>
              </div>
            )}
          </CardContent>
        </ScrollArea>
      )}
    </Card>
  );
}
