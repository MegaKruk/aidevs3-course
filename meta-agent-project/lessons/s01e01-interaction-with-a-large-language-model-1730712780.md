![](https://cloud.overment.com/S01E01-1730570331.png)

## Interaction with a Large Language Model

The ability of [large language models](glossary/LLM.md) to generate structured content allows their integration with application logic, enabling programmatic control over their behavior. At this stage of development, they act as tools that enable data processing and generation in ways previously impossible to programmatically achieve (e.g., with regex).

In AI_devs 3, we will focus on programmatic interaction with large language models via [APIs](glossary/LLM%20API.md), building partially autonomous tools called "[AI Agents](glossary/Agent.md)." These are complex solutions requiring practical programming experience and a good understanding of the nature of large language models.

These tools can perform a variety of tasks and processes but are not universal. Therefore, **we will focus on creating their individual components and modules.** This way, they can be combined in various configurations and adapted to our needs.

A few months ago, choosing a general-purpose model was practically limited to [OpenAI](services/OpenAI.md). Today, however, we can consider:

- [OpenAI](services/OpenAI.md): Models from the o1, GPT family, including TTS, Whisper, and Embedding
- [Anthropic](services/Anthropic.md): Claude family models (text + image only)
- [Vertex AI](services/Vertex%20AI.md) (Google): Gemini models and selected providers (e.g., Anthropic) and others
- [xAI](https://accounts.x.ai): Grok models, which quickly reached the top of the rankings (top10).
- [Amazon Bedrock](services/Amazon%20Bedrock.md) (Amazon): Models from Anthropic, Mistral, Meta, and others
- [Azure](services/Azure%20OpenAI%20Service.md) (Microsoft): OpenAI, Meta models, and others
- [Groq](services/Groq.md): Open Source models, such as Llama
- as well as several others, e.g., OpenRouter, Perplexity, Cerebras, Databricks, Mistral AI, or Together AI

We can therefore choose between different pricing offers, API access limits, privacy, and data processing policies, as well as the models themselves. This is important because [AI agents](glossary/Agent.md) will autonomously use our knowledge bases or gain access to tools. This will result in operating on a fairly large scale, considering even tens of millions of tokens, which generates noticeable costs. This is illustrated by the following example of a request to an AI Agent to save tasks in [Linear](https://linear.app/), which resulted in 17,400 input tokens and 461 output tokens. It is also worth noting the request execution time, which was "as long as" 24 seconds.

![Example request to AI Agent for task management illustrates the scale of processed tokens and response time](https://cloud.overment.com/2024-09-02/aidevs3_usage-c1dee228-3.png)

**One message, several actions taken, nearly 18,000 tokens, and half a minute for a response** — it sounds like a solution that doesn't make sense. However, let's look at it from a slightly different perspective.

Task management **requires active involvement from the person** operating a device with an application like Linear, Todoist, or ClickUp. The task must be named, described, assigned to a category, date, priority, or project **— this is how most of us work.**

![Application handling almost always requires direct human involvement controlling the entire process](https://cloud.overment.com/2024-09-02/aidevs3_human-1e0045b9-1.png)

This process can be partially automated. For example, we can use an API to monitor an email inbox and keywords appearing. Based on them, it is possible to set rules redirecting the message to a designated person or even create a new entry in a task application. **Here human involvement is not required, but a set of programmatically defined rules is needed (which is not always possible)** — thus, we are talking about process automation according to strictly defined rules.

![Private and business processes can be automated according to rigid rules, e.g., keyword matching](https://cloud.overment.com/2024-09-02/aidevs3_email-641f6178-2.png)

Now, we can use large language models to build such a system. Thanks to them, we can receive various forms of content from different sources since their interpretation and actions taken are handled by a model connected to application logic.

Such a system can receive data in the form of ordinary messages from another person, as well as through phone images, voice recordings from a watch, or messages sent on a laptop or from an external API. Moreover, the data source can even be **another AI agent!**

In the diagram below, we see that all data sources generate **a request**, which is interpreted by a large language model, on the basis of which an action plan is generated, and actions are taken.

![The combination of automation with a large language model allows quite free transformation of different content formats and dynamic adaptation to the situation](https://cloud.overment.com/2024-09-02/aidevs_agent-e32d845c-6.png)

What is especially interesting here is that during the execution of the above logic, it is possible to dynamically obtain access to information. For example, based on the mentioned project name, the agent can load additional information about it or fetch data from the Internet to enrich the description. On the other hand, if it can't handle the task ... it may ask for human help.

<div style="padding:75% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1005763540?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="0101_intro"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

And it is precisely with building such solutions that we will engage in the coming weeks, so — welcome to AI_devs 3!

## Connecting to the model, from a practical side

At this stage, I assume that you have already completed the AI_devs 3 introductory materials or are returning from previous editions. In both cases, you at least have a basic knowledge of language models. We can therefore move on to practical examples of interacting with models.

Let's start with the fact that by default, interaction with the model involves building an array of `messages` containing the conversation content combined with the system instruction, i.e., the [ChatML](glossary/ChatML.md) format. However, we are interested in a few additional issues.

Namely, the fact that generating a result in the case of an [AI Agent](glossary/Agent.md) consists of many queries and function calls. In the example below, we see 4 stages:

- **Understanding:** Requires loading memory and/or Internet access. This way, we go beyond the model's basic knowledge and gain information useful in later stages. It can be described as the "thinking" or "analysis" stage.
- **Action planning**: Involves combining previous "thoughts" with a list of available tools, skills, or other agents. Based on this, an action list is created to be performed in subsequent steps.
- **Taking action**: Requires knowledge, a plan, and available skills based on which the model decides on the next step and collects feedback.
- **Response**: Requires knowledge and a report from actions in order to generate the final answer.

![](https://cloud.overment.com/2024-09-02/aidevs3_plan-f0b12e52-d.png)

At this stage, it should be borne in mind that the above interaction **does not need to include human involvement** and can be carried out "in the background" and take from a few seconds to even several hours. It can also be launched automatically according to a schedule or external event.

We also see clearly that we are no longer talking about simple conversational building via a `messages` array, but a new architecture and application design patterns. Interestingly, this is programming that ~80% resembles classic applications, and [LLM](glossary/LLM.md), [Prompts](glossary/Prompt.md), or tools like [vector databases](glossary/Vector%20Database.md) play only a certain part. However, apart from the coding itself, significantly greater importance is attached to working with data, various file formats, organizing databases, or search strategies (so-called [retrieval](glossary/Retrieval.md)).

Returning to conducting interaction with the model, in the [`thread`](https://github.com/i-am-alice/3rd-devs/tree/main/thread) example, we see a rather unusual yet very useful way of conducting a conversation. Instead of sending the entire message history to the model each time, we apply **summarization** and only **the latest user message**. Thanks to this, we do not need a complete conversation for the model to remember key information such as the user's name.

To start this example, turn on the server with the command `bun thread` and make a GET request to `localhost:3000/api/demo`.

![](https://cloud.overment.com/2024-09-02/aidevs3_thread-676ee978-b.png)

The interaction scheme looks as follows: after providing the first response, a summary of the current conversation is generated, which is attached to the system prompt of the next turn. In this way, we transmit a "compressed" thread.

![](https://cloud.overment.com/2024-09-02/aidevs3_turns-faccf75e-d.png)

As a result of such compression, we naturally lose some information. However, nothing prevents adding a mechanism to search for previous threads in case the summary is insufficient.

The [`thread`](https://github.com/i-am-alice/3rd-devs/tree/main/thread) example is simple but perfectly illustrates how we can manipulate the course of the conversation and, as a result:

- The fact that we used a cheaper model for summarization is an example of **[cost optimization](glossary/cost%20reduction.md)**
- Thanks to summarization, the model processes a smaller amount of content, **thus its attention is more focused on the current task** — **IMPORTANT!** in GPT-4o class models, this is a very important issue affecting [model effectiveness](glossary/performance%20optimization.md)
- Summarization also allows avoiding the context window limit, which matters in the case of Open Source models that may be responsible for a selected interaction element (e.g., [anonymization](glossary/data%20anonymization.md))
- Summarization can also be used in other parts of the agent's logic as well as an element of the user interface or work report [agent](glossary/Agent.md)
- In this case, we used summarization, but the same scheme will be used for example for image, sound, or video recognition. There too, additional model queries will be used as part of the system prompt's context

## Types of interaction

The complex logic of [agents](glossary/Agent.md) consists of modules and individual actions. It is difficult to talk about a well-functioning system if we do not take care of details both on the prompt side and the code side. Therefore, at this stage, we will go through several examples of elementary actions, such as decision making, classification, parsing, transformation, and evaluation. We will also use the [PromptFoo](tools/PromptFoo.md) tool, the launch of which I discussed in lesson [S00E02 — Prompt Engineering](S00E02%20—%20Prompt%20Engineering.md) and which we will talk about more in future lessons, but for now, starting it will suffice.

An example of a single action could be **decision-making** by the model based on available data. It can be compared to an `if` condition or `switch`. The difference lies in flexibility at the cost of deterministic result.

The following scenario presents logic checking whether we need to use a search engine for a given request. At first glance, such a scenario suggests the use of [Function Calling / Tool Use](glossary/Function%20Calling.md). However, it will not always be obvious.

![](https://cloud.overment.com/2024-09-03/aidevs3_decision-a25871ac-c.png)

Namely, we assume here that we immediately have a complete set of necessary information needed to start the search, which is usually not true. Connecting, for example, with [FireCrawl](tools/FireCrawl.md) may require downloading a list of permissible domains or generating keywords based on additional context loaded from a database.
In the example [`use search`](https://github.com/i-am-alice/3rd-devs/tree/main/use_search) (requires [PromptFoo](tools/PromptFoo.md) installed), we have a prompt responsible for deciding whether to use a search engine. Its task is to generate `0` or `1` to classify a query. For this reason, I've included [Few-Shot](glossary/Few-Shot.md) examples and defined a set of rules tailored to initial assumptions. The operation of such a prompt is then automatically verified on dozens of examples. 

![](https://cloud.overment.com/2024-09-03/aidevs3_promptfoo-09e8b046-b.png)

Decision-making by [LLM](glossary/LLM.md) may involve selecting multiple options, not just one. In this case, we refer to query classification. Since we are talking about searching the Internet, a prompt that selects domains to narrow the search will also be helpful. This is useful because autonomous browsing of web pages quickly leads to low-quality sources or services that require logging in or blocking access to content.

Therefore, it is worth listing addresses and describing them so that LLM can decide when to include them and when not to. The prompt that accomplishes this task is in the example [`pick_domains`](https://github.com/i-am-alice/3rd-devs/tree/main/pick_domains). 

![](https://cloud.overment.com/2024-09-03/aidevs3_domains-08f186d6-5.png)

In the above prompt, to increase effectiveness, we also used the [Thought Generation](glossary/Thought%20Generation.md) variant, specifically "zero-shot chain of thought." We are talking about increasing effectiveness because this way we give [LLM](glossary/LLM.md) "time to think," which is visible in `o1` models by default. Additionally, the fact that the property "\_thoughts" is generated initially is related to the fact that language models are currently autoregressive, and the content of this first property affects subsequent content, thus increasing the probability of obtaining desired results. 

Staying on the topic of Internet searching, we are ready to make a query to the search engine. However, at this stage, we can only get results in a format known from Google or DuckDuckGo. This means they will not be sufficient information for providing the final answer, but we can indicate the pages we want to load based on them.

In the `rate` example, there is a prompt evaluating whether the returned result may contain information of interest to us. Based on the returned ratings, we will select the pages whose content we will want to load with the help, for example, of [FireCrawl](tools/FireCrawl.md).

![Automatic test of a prompt evaluating the importance of context from the query's point of view](https://cloud.overment.com/2024-09-04/aidevs_rate-7364bbb1-3.png)

Putting it all together, we already have:

- A decision on whether an internet search engine is needed
- A decision on what queries we want to direct to it
- The ability to filter returned results

So all that remains is to generate an answer to the original question based on the retrieved data. Let's see how we can combine all this into one in the example [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch). Its logic allows for a regular conversation with LLM, but when the need to use a search engine is detected, the user's original query is used to search the web, as can be seen in the video below.

<div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1006292055?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="01_01_websearch"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

![Example of a code connecting a large model with an internet search engine](https://cloud.overment.com/2024-09-04/aidevs3_websearch-f14ad4e3-1.png)

## Application Architecture

Even looking at the example [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch), one can notice that indeed ~80% of the code resembles a classic application. However, according to what we discussed in the [S00E04 — Programming](S00E04%20—%20Programowanie.md) lesson, natural language begins to appear in the code, and elements may have previously played a slightly lesser role depending on the project.

Besides directory structure, responsibility division, database architecture, or the technology stack itself, we must also consider the role of [large language models](glossary/LLM.md) and [prompts](glossary/Prompt.md). It's not just about choosing a model, hosting, or writing instructions, but primarily about data flow.

If we now look at the visualization of the example [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch), we can clearly see that the operation of successive prompts depends on the results of previous ones. Although we build each individually, when doing so, we must consider the data it will work on and how the data it generates will be used later. 

![](https://cloud.overment.com/2024-09-04/aidevs3_graph-f6782902-9.png)

A slightly more extended visualization shows these dependencies a bit more clearly. And yet, it is not everything because we are dealing with a [chain of prompts](glossary/Chain.md) and subsequent actions, which will not always be the case.

![](https://cloud.overment.com/2024-09-04/aidevs3_advanced-734782e6-2.png)

On the LangChain blog, you can read about [the basics of cognitive architecture](https://blog.langchain.dev/what-is-a-cognitive-architecture/), where the division into Code, LLM Call, Chain, Router, as well as State Machine and fully autonomous systems, is included, which we will still discuss. 

![](https://cloud.overment.com/2024-09-05/xnapper-2024-09-05-09.56.29-f389615d-4.png)

In the meantime, let's try to take a broader perspective, considering elements that we temporarily omitted to avoid high complexity. 

- **Database (e.g., [PostgreSQL](tools/PostgreSQL.md)):** now not only does the conversation history start anew every time. The content of search results and loaded pages also disappear after the request ends. So if the user asks a follow-up question in the next message, we will need to reload the same data. Thus, we see that **we will want to save not only the conversation history but also the context used to generate them**. The general mechanism is visible in the earlier example [`thread`](https://github.com/i-am-alice/3rd-devs/tree/main/thread), but it only considered the conversation content without additional context.
- **Search Engine (e.g., [Qdrant](tools/Qdrant.md)):** saving the mentioned data will quickly lead us to when loading them all becomes unprofitable or impossible. Key then will be their effective retrieval. We must think about how to organize and describe them effectively, then search and pass them to the model.
- **State Management:** similar to classic applications, here too, state management comes into play. However, here stored data will include prompt history or launched tool history along with feedback information.
- **API:** in the case of [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch), we have only two tools (web search and web scraping), but there will usually be many more. Each must be constructed in such a way that [LLM](glossary/LLM.md) can use it, understand responses, and handle errors.
- **Prompt Evaluation:** Looking at the above diagram, it becomes clear why earlier we went through [PromptFoo](tools/PromptFoo.md) and why we will modify prompts, testing them automatically on selected test sets.
- **Versioning and Backups:** versioning now refers not only to the history of code and prompts but also to changes made by the model. For example, a task list managing agent may accidentally modify entries that we do not want to edit, and we need an easy way to restore them.
- **Access Control:** in the example [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch) I programmatically limited the list of domains with which LLM can contact. Similarly, we will define model permissions to increase application stability.
- **Application Monitoring:** in the example [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch), I saved the history of executed queries in a markdown file. Naturally, this will not be sufficient in production applications where we will use [LangFuse](tools/LangFuse.md) or similar solutions for advanced monitoring.
- **Asynchronicity:** the [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch) tool shows that LLM can work in the background, thus extending response time. In such a case, it makes sense to run the script "in the background" or create a queue after which completion, the user will receive a notification or email about task completion.
- **Interface:** the [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch) tool can be used in a chat interface which I showed in the video. However, it could also be a form allowing you to add an address list and related tasks (e.g., "download the latest article") with a schedule for execution. 

We will cover all the above points in further lessons, but not all will be required every time. We will create both simple tools responsible for uncomplicated actions and extensive solutions supporting complex processes.

## Effectiveness Optimization

Instructions from the examples [`pick_domains`](https://github.com/i-am-alice/3rd-devs/tree/main/pick_domains), [`use_search`](https://github.com/i-am-alice/3rd-devs/tree/main/use_search), or [`rate`](https://github.com/i-am-alice/3rd-devs/tree/main/rate) contain from a few to several [Few-Shot](glossary/Few-Shot.md) examples. In some cases, there may even be dozens or hundreds, in which case we refer to "many-shot in-context learning," about which we can read in [Many-Shot In Context Learning](https://arxiv.org/abs/2404.11018).

![](https://cloud.overment.com/2024-09-05/aidevs3_manyshot-4edd19f3-b.png)

Including examples is the first technique we should consider when optimizing prompt effectiveness. By presenting expected behavior this way, we can reinforce the main instruction's content. 

We will deal with designing examples in further lessons, but for now, you need to know that:

- Examples typically take the form of pairs presenting input data (user message) and output data (model response)
- The number of examples usually does not exceed ~3 - 40 pairs
- Examples should present expected behavior, be diverse, and account for edge cases (e.g., those in which the model's behavior should differ from expected)
- Examples must be chosen carefully, but we can use model assistance to generate them
- Examples can eventually be used for [Fine-Tuning](glossary/Fine-Tuning.md), and their variants for automated tests
- A large number of examples can be combined with [a caching mechanism](https://www.anthropic.com/news/prompt-caching) to optimize costs and performance

We will deal with example selection techniques and working with them in the further part of AI_devs 3. Meanwhile, remember [Few-Shot](glossary/Few-Shot.md) as an inseparable element of practically every prompt. 
## Basics of Long-term Memory

Loading content from a search engine and web pages into the prompt to answer questions based on them is an example of [Retrieval-Augmented Generation](glossary/Retrieval-Augmented%20Generation.md). Here, with the help of [FireCrawl](tools/FireCrawl.md), we expanded the model's base knowledge exactly in the way we will do with knowledge bases or the agent's temporary and long-term memory. This is probably a familiar scheme, visible below, in which we indicate to the model the content it is to use when generating responses. 

![](https://cloud.overment.com/2024-09-05/aidevs3_rag-1d9ef809-b.png)
Analogously, we will load data from files, databases, or external services. However, the [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch) example showed us that searching for information will consist of several additional steps, involving query rephrasing, generating additional queries (so-called [Self-Querying](glossary/Self-Querying.md)), evaluating results (so-called [Re-rank](glossary/Re-rank.md)), and filtering them.

We must keep in mind that connecting external sources of knowledge to our system can have a very negative impact on its operation. This is precisely why I limited the list of domains for [FireCrawl](tools/FireCrawl.md), but there are more such situations. For instance, loading a PDF document may result in a loss of formatting, which will disrupt the understanding of its content.

Formatting issues are also not everything because we must constantly keep in mind LLM's limited knowledge of our context. For example, if we say "Remember, overment is my nickname", the system should remember "Adam's nickname is overment". Otherwise, it may later assume that 'overment' is its nickname, as shown in the example below.

![](https://cloud.overment.com/2024-09-05/aidevs3_nickname-b04b7bd3-c.png)

We will talk more about long-term memory for the model in the third AI_devs 3 module. For now, remember that:

- The quality of the model's statements depends on the prompt, but also on the data provided
- The instruction should contain information on how the model should use context in its statements
- One prompt can include multiple external contexts, but they should be clearly separated from each other
- The model should have instructions on what to do if the provided context is insufficient to answer
- We need to take care not only of the quality of the information sources provided but also of the method of storing and delivering it to the model. The aforementioned paraphrasing of memory shows that we must always ask ourselves the question: **How will the model use the knowledge provided?**

## Summary

This lesson is a foretaste of what we will be dealing with in the coming weeks. Its purpose was to show a broad perspective on applications using [LLM](glossary/LLM.md), and above all, that they are largely built in the same way as the software we create every day.

For this reason, one of the requirements for AI_devs 3 was the knowledge of at least one programming language. Although we will not be implementing all discussed elements ourselves and do not need experience in building databases or optimizing search engines, we will encounter various topics in the coming weeks that may be completely new to you. Therefore, treat this as an opportunity to experience a broad perspective on application development, and focus mainly on the areas that concern you (e.g., front-end, back-end, or databases).

After today's lesson, try at least to run the [`websearch`](https://github.com/i-am-alice/3rd-devs/tree/main/websearch) example and ask it a few questions to see how it handles them. There is a high probability that it will not effectively answer your questions — ponder why this happens. Go through the prompts from the `prompts.ts` file and check how effectively [FireCrawl](tools/FireCrawl.md) handles loading the contents of the pages you want to work with.

You can also spend some time working with [PromptFoo](tools/PromptFoo.md), whose basic configuration I discussed in the introductory material and lesson [S00E02 — Prompt Engineering](S00E02%20—%20Prompt%20Engineering.md). For working with this tool, use [Cursor IDE](tools/Cursor.md) with the loaded documentation, which will facilitate the generation of configuration files and faster comprehension of this solution.

Wrapping up today's lesson, it should already be clear to you that [LLM](glossary/LLM.md) in the application code allows for efficient natural language processing, as well as various data formats (e.g., audio or images). This gives us new possibilities, but the foundation of application development remains your knowledge and experience.