"use client"

import { Line, LineChart, CartesianGrid, XAxis, YAxis } from "recharts"

import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart"

const chartData = [
  { month: "January", desktop: 186, mobile: 80 },
  { month: "February", desktop: 305, mobile: 200 },
  { month: "March", desktop: 237, mobile: 120 },
  { month: "April", desktop: 73, mobile: 190 },
  { month: "May", desktop: 209, mobile: 130 },
  { month: "June", desktop: 214, mobile: 140 },
]

const chartConfig = {
  desktop: {
    label: "Desktop",
    color: "#2563eb",
  },
  mobile: {
    label: "Mobile",
    color: "#60a5fa",
  },
} satisfies ChartConfig

export function TopicChart() {
  return (
    <ChartContainer config={chartConfig} className="min-h-[200px] w-full">
      <LineChart accessibilityLayer data={chartData}>
        <ChartTooltip content={<ChartTooltipContent />} />
        <XAxis
          dataKey="month"
          tickLine={false}
          tickMargin={10}
          axisLine={true}
          tickFormatter={(value) => value.slice(0, 3)}
        />
        <YAxis label={{ position: "insideBottomRight" }} />
        <ChartLegend content={<ChartLegendContent />} />
        <CartesianGrid vertical={false} />
        <Line dataKey="desktop" fill="var(--color-desktop)" radius={4} />
        <Line dataKey="mobile" fill="var(--color-mobile)" radius={4} />
      </LineChart>
    </ChartContainer>
  )
}
