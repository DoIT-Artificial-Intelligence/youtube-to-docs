**How youtube-to-docs works**

youtube-to-docs turns videos into documents to search, share and hear. Here's how it works, in six steps.

**1. Point it at YouTube**

Start with a video link, a video ID, a playlist or a whole channel. Playlists and channels are expanded, and each video runs in turn.

**2. Get the words**

Next, it gets the words. By default, it uses YouTube's own captions. Or it transcribes the audio again with Gemini, Google Cloud or AWS. That gives real word timings and speaker labels. Either way, you get a transcript and a timed SRT file.

**3. Let the model read**

Then a language model of your choice reads the transcript. It can run on Gemini, Vertex AI, AWS Bedrock or Azure Foundry. It names the speakers and their titles. It pairs each question with its answer and a timestamp link. It writes a summary, a one-sentence summary and topic tags. Gemini can also watch the video, to read slides and name plates.

**4. Turn it into media**

From the summary, it can draw an infographic. Another model looks at the image and writes its alt text. It can read the summary aloud, and combine both into a video. And it can make everything again in a second language.

**5. Make the video accessible**

Add the -cc flag to make the original video accessible. Captions are corrected, and each speaker is named. They meet WCAG 2.1 Level AA and Section 508. Slides and on-screen text are described for blind viewers.

**6. You stay in control**

Every output is a file that you can review and edit. Run it again, and the later steps use your edits. A CSV lists every file and what each step cost. Save it all to your computer, Google Drive, SharePoint or Hugging Face.

**Get started**

Run it from the command line, the web app, or as a plugin for your AI assistant. One command takes you from a link to a full set of documents.
