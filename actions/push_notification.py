import cv2
import asyncio
import cloudinary
import cloudinary.uploader
from aiohttp import ClientSession
from concurrent.futures import ThreadPoolExecutor

cloudinary.config(cloud_name='dg7qyezkd', api_key='413219632115614',
                  api_secret='dqMcml_QHfjMvJPtxLvNHJ6PG_M')


def upload_to_cloudinary():
    return cloudinary.uploader.upload('temp.png')


async def upload_wrapper():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(ThreadPoolExecutor(), upload_to_cloudinary)


async def push(title='Se ha generado una alerta', description=''):
    ret = await upload_wrapper()
    async with ClientSession() as session:
        data = {
            'title': title,
            'description': description,
            'image_url': ret['url']
        }

        # Send data to push notification server
        async with session.post('http://localhost:3000/push_notification', json=data) as resp:
            if resp.status != 200:
                print('ERROR: Error sending data to push notification server')
            else:
                print(
                    f'Succesfully sent push notification, status = {resp.status}')

async def call():
    await asyncio.gather(push())

def push_notification(frame):
    cv2.imwrite('temp.png', frame)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(call())