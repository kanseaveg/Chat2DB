"""
    HttpClient Common Class
"""
import requests


class HttpClient:
    def __init__(self, headers=None, timeout=None):
        self.headers = headers
        self.timeout = timeout

    async def post(self, url, data=None, json=None):
        response = requests.post(
            url,
            data=data,
            json=json,
            headers=self.headers,
            timeout=self.timeout
        )
        return response.text

    async def post_with_proxy(self, url, data=None, json=None, proxy=None):
        response = requests.post(
            url,
            data=data,
            json=json,
            headers=self.headers,
            timeout=self.timeout,
            # proxies={'http': proxy, 'https': proxy},
        )
        return response.text
    
    async def get(self, url, params=None):
        response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
        )
        return response.text
            
