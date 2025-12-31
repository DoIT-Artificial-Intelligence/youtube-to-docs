Currently this repo processes data locally as .csv/.md files.

I want to make it so it you also use "workspace" as a back-end (Google Workspace). i.e. a user can say "workspace" to the argument -o and it will sync files to the user's Google Drive. By default it will create a folder called (youtube-to-docs-artifacts) and add the relevant folders in here. The youtube-docs.csv will now be a google sheet called youtube-docs. the text and markdown files will now be google docs. in the output file the file paths will be the associated google workspace file path e.g. "https://docs.google.com/document/d/{DOCUMENT_ID}".

See tests/test_workspace.py for an example of how to upload a file to Google Drive.

In the future the -o argument can also take "M365" as an argument that will do the same thing but with Microsoft (word files and excel) but's let walk before we run and test things along the way.

The development plan should be:
 - [] Refactor the code base so that if the arguments to -o are "workspace" or "GOOGLE FOLDER ID" you check if any files are created in there and use those.
     Add a note to "development.md" for clearing the cache of these files when developing this project
 - [] Run commands to test works as expected. Use L6smIHfRjNk as it's a short video (27:39):
   - [] `uv run python -m youtube_to_docs.main L6smIHfRjNk`  # This should upload a YouTube generated transcript file to my Google Drive in a folder called youtube-docs-articacts/transcript-files. It should also create a google sheet called "youtube-docs" in the youtube-docs-articacts folder which contains the url's for any artifacts generated
   - [] `uv run python -m youtube_to_docs.main L6smIHfRjNk -t gemini-3-flash-preview`  # This should create a file in the folder youtube-docs-articacts/transcript-files. It should also add a m4a file in the folder youtube-docs-articacts/audio-files and update the google sheet "youtube-docs"
   - [] `uv run python -m youtube_to_docs.main L6smIHfRjNk -m bedrock-nova-2-lite-v1`  # This should create two summary files in the folder youtube-docs-articacts/summary-files and update the google sheet. It should also create files in speaker-extraction-files/ and qa-files/ and update the google sheet
   - [] `uv run python -m youtube_to_docs.main L6smIHfRjNk -i gemini-2.5-flash-image`  # This should create two infographics in the folder youtube-docs-articacts/infographic-files and update the google sheet.
   - [] `uv run python -m youtube_to_docs.main L6smIHfRjNk --tts gemini-2.5-flash-preview-tts-Kore`  # This should create two audio files in the folder youtube-docs-articacts/audio-files
   - [] Delete the youtube-docs-articacts workspace folder
   - [] `uv run python -m youtube_to_docs.main L6smIHfRjNk -t gemini-3-flash-preview -m bedrock-nova-2-lite-v1 -i gemini-2.5-flash-image --tts gemini-2.5-flash-preview-tts-Kore -l es --no-youtube-summary and checks that files are created as expected.



