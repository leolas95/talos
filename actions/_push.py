import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cloudinary
import cloudinary.uploader
from aiohttp import ClientSession

cloudinary.config(cloud_name='dg7qyezkd', api_key='413219632115614',
                  api_secret='dqMcml_QHfjMvJPtxLvNHJ6PG_M')

PUSH_NOTIFICATION_ENDPOINT = 'https://push-notifications-server.herokuapp.com/push_notification'

def upload_to_cloudinary():
    return cloudinary.uploader.upload('temp.png')


async def upload_wrapper():
    upload_loop = asyncio.get_event_loop()
    return await upload_loop.run_in_executor(ThreadPoolExecutor(), upload_to_cloudinary)


async def push(title, description):
    ret = await upload_wrapper()
    async with ClientSession() as session:
        data = {
            'title': title,
            'description': description,
            'image_url': ret['url']
        }

        # Send data to push notification server
        async with session.post(PUSH_NOTIFICATION_ENDPOINT, json=data) as resp:
            if resp.status != 200:
                print('ERROR: Error sending data to push notification server')
            else:
                print(
                    f'Succesfully sent push notification, status = {resp.status}')
            os.remove('temp.png')


async def call(title='Se ha generado una alerta', description=''):
    await asyncio.gather(push(title, description))


def main():
    title = sys.argv[1]
    description = sys.argv[2]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(call(title, description))


if __name__ == '__main__':
    main()
