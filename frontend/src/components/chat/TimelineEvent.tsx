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
  if (title.toLowerCase().includes('generating')) {
    return <TextSearch className="h-4 w-4 text-neutral-400" />;
  } else if (title.toLowerCase().includes('thinking')) {
    return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
  } else if (title.toLowerCase().includes('reflection')) {
    return <Brain className="h-4 w-4 text-neutral-400" />;
  } else if (title.toLowerCase().includes('research')) {
    return <Search className="h-4 w-4 text-neutral-400" />;
  } else if (title.toLowerCase().includes('finalizing')) {
    return <Pen className="h-4 w-4 text-neutral-400" />;
  }
  return <Activity className="h-4 w-4 text-neutral-400" />;
};

const TimelineEvent: React.FC<TimelineEventProps> = ({ event, isLast, isLoading }) => {
  const renderData = (data: any) => {
    if (typeof data === 'string') {
      return data;
    }
    if (Array.isArray(data)) {
      return data.join(', ');
    }
    return JSON.stringify(data);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
      className="relative pl-8 pb-4"
    >
      {!isLast || (isLoading && isLast) ? (
        <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-600" />
      ) : null}
      <div className="absolute left-0.5 top-2 h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
        {getEventIcon(event.title)}
      </div>
      <div>
        <p className="text-sm text-neutral-200 font-medium mb-0.5">{event.title}</p>
        <p className="text-xs text-neutral-300 leading-relaxed">
          {renderData(event.data)}
        </p>
      </div>
    </motion.div>
  );
};

export default TimelineEvent;
