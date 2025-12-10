"use client"

import { TopicChart } from "./TopicChart"
import { SentimentChart } from "./SentimentChart"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { useState } from "react"

export default function Home() {
  const [showTopicChart, setShowTopicChart] = useState(true)

  console.log(showTopicChart)

  return (
    <>
      <RadioGroup
        defaultValue="topic"
        onValueChange={() => setShowTopicChart(!showTopicChart)}
      >
        <h2>Select chart to show: </h2>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="topic" id="topic" />
          <Label htmlFor="topic">Topic trends chart</Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="sentiment" id="sentiment" />
          <Label htmlFor="sentiment">Sentiment trends chart</Label>
        </div>
      </RadioGroup>
      <div className="max-w-300 p-10">
        {showTopicChart ? <TopicChart /> : <SentimentChart />}
      </div>
    </>
  )
}
