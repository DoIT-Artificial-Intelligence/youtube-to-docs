# Maryland bill hearing videos

The Maryland General Assembly (MGA) streams its floor sessions and committee hearings on YouTube. Each committee has its own channel. This page lists them so you can point `youtube-to-docs` at the right one.

The official list is on the MGA website: [Senate](https://mgaleg.maryland.gov/mgawebsite/Information/YouTubeChannelInfo?id=s), [House](https://mgaleg.maryland.gov/mgawebsite/Information/YouTubeChannelInfo?id=h) and [Other](https://mgaleg.maryland.gov/mgawebsite/Information/YouTubeChannelInfo?id=o). The tables below match that list as of November 2026.

## Senate

| Committee | Channel | Uploads playlist |
| :--- | :--- | :--- |
| Senate Floor | [@mga-session-senate445](https://www.youtube.com/@mga-session-senate445) | `UU965i3n0wndwj90muUZlyFQ` |
| Budget and Taxation (B&T) | [@mga-bt5827](https://www.youtube.com/@mga-bt5827) | `UU3Jf9teuxVj0ezUE6C9hf-w` |
| Education, Energy, and the Environment (EEE) | [@mga-eee1773](https://www.youtube.com/@mga-eee1773) | `UUYqghFoT_BDQ0N5xjvTacgg` |
| Executive Nominations (EXN) | [@mga-exn82](https://www.youtube.com/@mga-exn82) | `UU6-BUQh72s7qc5dYHi4DGrA` |
| Finance (FIN) | [@mga-fin9827](https://www.youtube.com/@mga-fin9827) | `UUJ6eORINhf8eESejcQiItYw` |
| Judicial Proceedings (JPR) | [@mga-jpr6516](https://www.youtube.com/@mga-jpr6516) | `UUxtIdwSbwpptrfW5AOwtqqg` |
| Rules (SRU) | [@mga-sru2426](https://www.youtube.com/@mga-sru2426) | `UUXneuDdFBxryssX80hpGlsA` |
| Senate Other Committees (subcommittees and others) | [@mga-senateother6127](https://www.youtube.com/@mga-senateother6127) | `UUzGXZIUul2VLKHDFVkXlKBA` |

## House of Delegates

| Committee | Channel | Uploads playlist |
| :--- | :--- | :--- |
| House Floor | [@mga-session-house9230](https://www.youtube.com/@mga-session-house9230) | `UUqELNxEkj2ZNATl4eXWouRw` |
| Appropriations (APP) | [@mga-app8026](https://www.youtube.com/@mga-app8026) | `UUvZWgCoEP1pZcE1QvnfPk2A` |
| Economic Matters (ECM) | [@mga-ecm6367](https://www.youtube.com/@mga-ecm6367) | `UUYOYQq7WvSGQQI_UZdv52HA` |
| Environment and Transportation (ENT) | [@mga-ent2368](https://www.youtube.com/@mga-ent2368) | `UUpsNhxGJ7NFdwRjSZFyg94Q` |
| Government, Labor, and Elections (GLE) | [@mga-gle2536](https://www.youtube.com/@mga-gle2536) | `UUffkX_pcQTcz1E--BmFK7Jw` |
| Health (HLT) | [@mga-hlt2698](https://www.youtube.com/@mga-hlt2698) | `UUasYvXf7VyHfNt-RjCq-PnQ` |
| Judiciary (JUD) | [@mga-jud9614](https://www.youtube.com/@mga-jud9614) | `UUljsiF5QnMYgaAU6CCuXOjg` |
| Ways and Means (W&M) | [@mga-wm8116](https://www.youtube.com/@mga-wm8116) | `UUm2qzgHptepCfoKxzZEiW2w` |
| Rules and Executive Nominations (HRU) | [@mga-hru3883](https://www.youtube.com/@mga-hru3883) | `UUXMvaMeKlm7d_J1aMIX3uEQ` |
| House Other Committees (subcommittees and others) | [@mga-houseother6347](https://www.youtube.com/@mga-houseother6347) | `UUNDsfDhn7AYOcQYuI2JZm2w` |

## Joint and other committees

| Committee | Channel | Uploads playlist |
| :--- | :--- | :--- |
| Other Committees (joint committees, task forces, Spending Affordability and others) | [@mga-othercommittees6625](https://www.youtube.com/@mga-othercommittees6625) | `UU98TvUSKpiFdVg8fkvrVMGA` |

## Older channels

These channels are not on the official list, but they still hold older meetings.

| Channel | What it has |
| :--- | :--- |
| [@mga-hgo1740](https://www.youtube.com/@mga-hgo1740) | Health and Government Operations (HGO) sessions and some other House committees |
| [@mgalegislature2598](https://www.youtube.com/@mgalegislature2598) | Virtual meetings and witness testimony from 2020 and early 2021 |
| [@mgalegislaturechannel1652](https://www.youtube.com/@mgalegislaturechannel1652), [@mgalegislaturechannel265](https://www.youtube.com/@mgalegislaturechannel265), [@mgalegislaturechannel397](https://www.youtube.com/@mgalegislaturechannel397), [@mgalegislaturechannel4981](https://www.youtube.com/@mgalegislaturechannel4981) | Virtual briefings, task forces and joint committee meetings from 2020 and early 2021 |

## Find the hearing for a bill

1. Open the bill on the [MGA website](https://mgaleg.maryland.gov/mgawebsite/Search/Legislation), for example [HB0001 (2026)](https://mgaleg.maryland.gov/mgawebsite/Legislation/Details/hb0001?ys=2026RS).
2. In the committee section, note the committee and the hearing date.
3. Open that committee's channel above. Videos are titled with the committee code and date, for example `JUD Committee Session, 4/11/2026 #1` or `House Floor Session, 11/19/2026 #10`. A `#2` or higher means more than one session that day.
4. Copy the video URL and run `youtube-to-docs` on it.

To see what is coming up, use the MGA [meeting schedule](https://mgaleg.maryland.gov/mgawebsite/Meetings/Day). To browse past recordings by year, use [Search Media](https://mgaleg.maryland.gov/mgawebsite/Search/Media).

!!! note
    The bill page also has a **Click to view Recorded Media** link. It opens the MGA's own video player, not YouTube. `youtube-to-docs` takes YouTube input only, so use the committee's YouTube channel instead.

Most hearings are posted as live streams. Live streams show under the channel's **Live** tab, not **Videos**. `youtube-to-docs` includes both.

## Process hearings

Process one hearing:

```bash
youtube-to-docs https://www.youtube.com/watch?v=VIDEO_ID -m -i
```

Process a whole channel (this can be hundreds of videos):

```bash
youtube-to-docs @mga-jud9614 -m
```

Process only the newest hearings. Use the uploads playlist ID from the tables above and add `n=` with the number of videos you want. This runs the 5 newest Judiciary hearings:

```bash
youtube-to-docs UUljsiF5QnMYgaAU6CCuXOjgn=5 -m
```

Channel handles and playlists need a `YOUTUBE_DATA_API_KEY`. See [Environment variables](usage.md#environment-variables) for setup and [Usage](usage.md) for all options.

!!! tip
    A channel's uploads playlist ID is its channel ID with the leading `UC` changed to `UU`.
