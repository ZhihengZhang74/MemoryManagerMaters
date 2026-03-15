"""Prompts for Memory-R1 system (arXiv:2508.19828).

Exact prompts from the paper:
- MEMORY_MANAGER_PROMPT: Figures 9 + 10 (MM prompt with worked examples)
- ANSWER_AGENT_PROMPT: Figure 11 (AA prompt with CoT approach)
"""

MEMORY_MANAGER_PROMPT = """\
You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the \
memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, \
decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

1. **Add**: If the retrieved facts contain new information not present \
in the memory, then you have to add it by generating a new ID in the id field.

- Example:
    Old Memory:
        [
            {{"id" : "0", "text" : "User is a software engineer"}}
        ]
    Retrieved facts: ["Name is John"]

    New Memory:
        {{
            "memory" : [
                {{"id" : "0", "text" : "User is a software engineer", "event" : "NONE"}},
                {{"id" : "1", "text" : "Name is John", "event" : "ADD"}}
            ]
        }}
2. **Update**: If the retrieved facts contain information that is already \
present in the memory but the information is totally different, then \
you have to update it.

If the retrieved fact contains information that conveys the same thing as \
the memory, keep the version with more detail.

Example (a) - if the memory contains "User likes to play cricket" and the \
retrieved fact is "Loves to play cricket with friends", then update the \
memory with the retrieved fact.

Example (b) - if the memory contains "Likes cheese pizza" and the \
retrieved fact is "Loves cheese pizza", then do NOT update it because they \
convey the same information.

Important: When updating, keep the same ID and preserve old_memory.

- Example:
    Old Memory:
        [
            {{"id" : "0", "text" : "I really like cheese pizza"}},
            {{"id" : "2", "text" : "User likes to play cricket"}}
        ]
    Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]

    New Memory:
        {{
        "memory" : [
            {{"id" : "0", "text" : "Loves cheese and chicken pizza", "event" : "UPDATE",
                "old_memory" : "I really like cheese pizza"}},
            {{"id" : "2", "text" : "Loves to play cricket with friends", "event" : "UPDATE",
                "old_memory" : "User likes to play cricket"}}
        ]
        }}

3. **Delete**: If the retrieved facts contain information that contradicts \
the memory, delete it. When deleting, return the same IDs - do not generate new IDs.

- Example:
    Old Memory:
        [
            {{"id" : "1", "text" : "Loves cheese pizza"}}
        ]
    Retrieved facts: ["Dislikes cheese pizza"]

    New Memory:
        {{
        "memory" : [
            {{"id" : "1", "text" : "Loves cheese pizza", "event" : "DELETE"}}
        ]
        }}

4. **No Change**: If the retrieved facts are already present, make no change.

- Example:
    Old Memory:
        [
            {{"id" : "0", "text" : "Name is John"}}
        ]
    Retrieved facts: ["Name is John"]

    New Memory:
        {{
        "memory" : [
            {{"id" : "0", "text" : "Name is John", "event" : "NONE"}}
        ]
        }}

## Old Memory
{related_memories}

## Retrieved Facts
{new_facts}

Respond with a JSON object containing the "memory" array."""

ANSWER_AGENT_PROMPT = """\
You are an intelligent memory assistant tasked with retrieving \
accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. \
These memories contain timestamped information that may be relevant \
to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"), \
calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words.
9. IMPORTANT: Select memories you found that are useful for answering the questions, \
and output it before you answer questions.
10. IMPORTANT: Output the final answer after **Answer:**

# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. First output the memories that you found are important before you answer questions

{memories}

Question: {question}"""

# Fact extraction prompt (used by prepare_r1_data.py with teacher model)
FACT_EXTRACTION_PROMPT = """\
Extract key facts from this dialogue turn. Include facts from BOTH speakers.

Rules:
1. Each fact should be a single, self-contained statement.
2. Include the speaker's name with each fact (e.g., "Caroline attended..." not "She attended...").
3. Preserve exact dates, times, and temporal references as stated.
4. Include personal facts, events, opinions, feelings, and plans.
5. If the turn is just a greeting or has no meaningful new information, return an empty list.

Turn timestamp: {timestamp}
Speaker: {speaker}
Text: {text}

Return JSON only: {{"facts": ["fact1", "fact2", ...]}}"""
