"use client";

import * as React from "react";
import { LineChart, Line, CartesianGrid, XAxis, YAxis } from "recharts";

import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
    ChartLegend,
    ChartLegendContent,
    type ChartConfig,
} from "@/components/ui/chart";

//input is 2025-03-08 iso string format
function formatWeek(iso: string) {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

//render small popup for the hovered topic
function SingleTopicTooltip({
    active,
    payload,
    hoveredTopic,
    sentimentMatrix,
}: {
    active?: boolean; //true when to render
    payload?: any[]; //data for all topics at the hovered week
    hoveredTopic: string | null; //which topic is hovered
    sentimentMatrix: (number | null)[][];
}) {
    //do nothing
    if (!active || !payload?.length || !hoveredTopic) return null;

    //takes the data for the hovered topic
    const item = payload.find((p) => p.dataKey === hoveredTopic);
    if (!item) return null;

    //get index from topic_index and make it into a number
    const topicIdx = Number(hoveredTopic.replace("topic_", ""));

    // get week index by finding the hovered row in the chart
    const weekIso: string = item.payload.week;
    const weekIdx = Number(item.payload.__weekIndex ?? -1); //added to the weekly topic counts json
    const resolvedWeekIdx = weekIdx >= 0 ? weekIdx : undefined;

    const sentiment =
        //index sentimemt matrix with week and topic index if it is defined and valid -1 is invalid
        resolvedWeekIdx !== undefined && resolvedWeekIdx >= 0
            ? sentimentMatrix?.[resolvedWeekIdx]?.[topicIdx]
            : null;

    return (
        <div className="rounded-md border bg-background px-2.5 py-1.5 text-xs shadow">
            <div className="font-medium">{item.name}</div>
            <div className="font-mono tabular-nums">
                {item.value.toLocaleString()} posts
            </div>

            {/*sentiment in pop-up */}
            <div className="font-mono tabular-nums">
                Sentiment:{" "}
                {sentiment === null || sentiment === undefined
                    ? "N/A"
                    : sentiment.toFixed(3)}
            </div>
            <div className="text-muted-foreground">{weekIso}</div>
        </div>
    );
}

export function TopicChart() {
    const [data, setData] = React.useState<any[]>([]);
    const [summaries, setSummaries] = React.useState<string[]>([]);
    const [hoveredTopic, setHoveredTopic] = React.useState<string | null>(null);
    const [sentimentMatrix, setSentimentMatrix] = React.useState<
        (number | null)[][]
    >([]);

    //these run once to load data

    React.useEffect(() => {
        fetch("/sentimented_posts_per_week_per_cluster.json")
            .then((r) => r.json())
            .then(setSentimentMatrix)
            .catch(console.error);
    }, []);

    // Load chart data
    React.useEffect(() => {
        fetch("/weekly_topic_counts.json")
            .then((r) => r.json())
            .then(setData);
    }, []);

    // Load topic summaries
    React.useEffect(() => {
        fetch("/topic_cluster_texts.json")
            .then((r) => r.json())
            .then(setSummaries);
    }, []);

    const { config, topicKeys } = React.useMemo(() => {
        const first = data?.[0];
        const keys = first
            ? Object.keys(first).filter((k) => k.startsWith("topic_"))
            : [];

        const palette = [
            "hsl(221 83% 53%)",
            "hsl(142 71% 45%)",
            "hsl(0 84% 60%)",
            "hsl(43 96% 56%)",
            "hsl(262 83% 58%)",
            "hsl(199 89% 48%)",
        ];

        const cfg: ChartConfig = {};
        keys.forEach((k, i) => {
            cfg[k] = {
                label: `Topic ${i}`,
                color: palette[i % palette.length],
            };
        });

        return { config: cfg, topicKeys: keys };
    }, [data]);

    if (!data.length) return <div>Loading…</div>;

    const hoveredIndex = hoveredTopic
        ? Number(hoveredTopic.replace("topic_", ""))
        : null;
    const hoveredSummary =
        hoveredIndex !== null ? summaries[hoveredIndex] : null;

    return (
        <div className="space-y-4">
            <ChartContainer config={config} className="h-[420px] w-full">
                <LineChart data={data}>
                    <CartesianGrid vertical={false} />
                    <XAxis
                        dataKey="week"
                        tickFormatter={formatWeek}
                        tickLine={false}
                        axisLine={false}
                    />
                    <YAxis
                        allowDecimals={false}
                        tickLine={false}
                        axisLine={false}
                    />
                    <ChartTooltip
                        content={(props) => (
                            <SingleTopicTooltip
                                {...props}
                                hoveredTopic={hoveredTopic}
                                sentimentMatrix={sentimentMatrix}
                            />
                        )}
                    />
                    <ChartLegend content={<ChartLegendContent />} />
                    {topicKeys.map((key) => {
                        const isActive = hoveredTopic === key;
                        const isDimmed = hoveredTopic && !isActive;

                        return (
                            <Line
                                key={key}
                                type="monotone"
                                dataKey={key}
                                stroke={`var(--color-${key})`}
                                strokeWidth={isActive ? 3 : 2}
                                strokeOpacity={isDimmed ? 0.2 : 1}
                                dot={false}
                                onMouseEnter={() => setHoveredTopic(key)}
                                onMouseLeave={() => setHoveredTopic(null)}
                            />
                        );
                    })}
                </LineChart>
            </ChartContainer>

            {/*Summary box */}
            <div className="rounded-lg border bg-muted/30 p-4 text-sm min-h-[4rem]">
                {hoveredSummary ? (
                    <>
                        <div className="font-medium mb-1">
                            {config[hoveredTopic!]?.label}
                        </div>
                        <p className="text-muted-foreground leading-relaxed">
                            {hoveredSummary}
                        </p>
                    </>
                ) : (
                    <span className="text-muted-foreground">
                        Hover over a topic line to see its summary :D
                    </span>
                )}
            </div>
        </div>
    );
}
