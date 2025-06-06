"use client"

import * as React from "react"
import * as SwitchPrimitives from "@radix-ui/react-switch"
import { cva, type VariantProps } from "class-variance-authority" // cva might not be strictly necessary here but often included

import { cn } from "@/lib/utils"

// Base styling for the switch, derived from typical ShadCN setups.
// It uses data-state attributes for checked/unchecked states.
const switchRootVariants = cva(
  "peer inline-flex h-[24px] w-[44px] shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=unchecked]:bg-input",
  {
    variants: {}, // No specific variants defined here, but structure allows for it
    defaultVariants: {},
  }
)

const switchThumbVariants = cva( // Also define thumb variants if you want to customize it via cva
    "pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0"
)


const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root> & VariantProps<typeof switchRootVariants>
>(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    className={cn(switchRootVariants(), className)}
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb className={cn(switchThumbVariants())} />
  </SwitchPrimitives.Root>
))
Switch.displayName = SwitchPrimitives.Root.displayName

export { Switch }
