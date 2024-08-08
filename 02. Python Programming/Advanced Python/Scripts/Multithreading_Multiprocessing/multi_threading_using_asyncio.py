# using asyncio
import asyncio
import time
import aiohttp
from logging import basicConfig, info, INFO


DO_NOT_ABUSE_THE_SERVER = 40


async def download_site(session, url):
    async with session.get(url) as response:
        print("Read {0} from {1}".format(response.content_length, url))


async def download_all_sites(sites):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in sites:
            task = asyncio.ensure_future(download_site(session, url))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":

    # logging configuration
    format = "%(levelname)s - %(message)s"
    basicConfig(
        filename="logging_result.log",
        level=INFO,
        format=format,
        filemode="a",
    )

    sites = [
        "https://www.jython.org",
        "https://devguide.python.org/internals/exploring/",
    ] * DO_NOT_ABUSE_THE_SERVER

    start_time = time.time()
    asyncio.get_event_loop().run_until_complete(download_all_sites(sites))
    duration = time.time() - start_time

    info(
        f"Multi-threading with asyncio downloaded {len(sites)} sites in {duration:.2f} seconds"
    )

    print(f"Downloaded {len(sites)} sites in {duration} seconds")
