import { motion } from "framer-motion";
import { cn } from "@/lib/utils"; // Assuming cn is available

export function LoadingBubble() {
  const dotVariants = {
    initial: {
      opacity: 0.5,
      scale: 0.8,
    },
    animate: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.5,
        repeat: Infinity,
        repeatType: "reverse",
        ease: "easeInOut",
      },
    },
  };

  return (
    <motion.div
      className={cn(
        "flex items-end gap-3 justify-start", // Mimic MessageBubble alignment for AI
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
    >
      <div
        className={cn(
          "relative break-words flex flex-row items-center space-x-1.5 rounded-xl min-h-7 max-w-fit p-4 shadow-sm", // Use flex-row for dots
          "bg-card text-card-foreground rounded-bl-md" // AI bubble style
        )}
      >
        {[0, 1, 2].map((index) => (
          <motion.div
            key={index}
            className="h-2.5 w-2.5 bg-primary rounded-full" // Use primary color for dots
            variants={dotVariants}
            initial="initial"
            animate="animate"
            custom={index}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "easeInOut",
              delay: index * 0.2, // Stagger the animation
            }}
          />
        ))}
      </div>
    </motion.div>
  );
}
