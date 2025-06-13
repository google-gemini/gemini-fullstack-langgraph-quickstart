import React from 'react';
import { motion } from 'framer-motion';
import { Loader2, Activity, Search, TextSearch, Brain, Pen } from 'lucide-react';
import type { ProcessedEvent } from '../ActivityTimeline';

interface TimelineEventProps {
  event: ProcessedEvent;
  isLast: boolean;
  isLoading: boolean;
}

const getEventIcon = (title: string) => {
  const iconClasses = "h-4 w-4 text-muted-foreground"; // Updated icon color
  if (title.toLowerCase().includes('generating')) {
    return <TextSearch className={iconClasses} />;
  } else if (title.toLowerCase().includes('thinking')) {
    return <Loader2 className={cn(iconClasses, "animate-spin")} />;
  } else if (title.toLowerCase().includes('reflection')) {
    return <Brain className={iconClasses} />;
  } else if (title.toLowerCase().includes('research')) {
    return <Search className={iconClasses} />;
  } else if (title.toLowerCase().includes('finalizing')) {
    return <Pen className={iconClasses} />;
  }
  return <Activity className={iconClasses} />;
};

// Helper to import cn for getEventIcon if it's not already available globally in this file.
// For this exercise, assuming cn is available or this would be part of a broader setup.
import { cn } from "@/lib/utils";

const TimelineEvent: React.FC<TimelineEventProps> = ({ event, isLast, isLoading }) => {
  const renderData = (data: any) => {
    if (typeof data === 'string') {
      return data;
    }
    if (Array.isArray(data)) {
      // Make array data more readable, perhaps one item per line if it's too long
      if (data.join(', ').length > 100) {
        return (
          <ul className="list-disc list-inside">
            {data.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        );
      }
      return data.join(', ');
    }
    // For objects, consider a more structured display if they are not too complex
    if (typeof data === 'object' && data !== null) {
        const keys = Object.keys(data);
        if (keys.length > 0 && keys.length < 5 && JSON.stringify(data).length < 200) {
             return (
                <ul className="text-xs space-y-0.5">
                {keys.map(key => (
                    <li key={key}><span className="font-medium">{key}:</span> {String(data[key])}</li>
                ))}
                </ul>
             );
        }
    }
    return JSON.stringify(data);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.25, ease: "easeOut" }} // Updated animation
      className="relative pl-10 pb-5" // Increased left padding for icon, increased bottom padding
    >
      {!isLast || (isLoading && isLast) ? (
        <div className="absolute left-[18px] top-[22px] h-full w-0.5 bg-border" /> // Adjusted position for new icon size/padding
      ) : null}
      <div className="absolute left-0 top-3 h-9 w-9 rounded-full bg-muted flex items-center justify-center ring-2 ring-background">
        {/* Updated: bg-muted, ring-2, ring-background (to blend with card bg), increased size slightly */}
        {getEventIcon(event.title)}
      </div>
      <div className="ml-2"> {/* Added small margin for content separation from icon line */}
        <p className="text-sm text-foreground font-medium mb-1">{event.title}</p> {/* Updated text color and margin */}
        <p className="text-xs text-muted-foreground leading-snug"> {/* Updated text color and leading */}
          {renderData(event.data)}
        </p>
      </div>
    </motion.div>
  );
};

export default TimelineEvent;
