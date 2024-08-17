"""
    Use a POST request for the single-turn Text-to-SQL model over HTTP.
"""
import json
import logging


async def nl2sql(state, q_s):
    settings = state.settings
    client = state.client
    url = settings.ATTENTION_BASED_SINGLE_TURN_SERVICE_ADDRESS
    try:
        response = await client.post(
            url,
            json={
                "entry": q_s
            }
        )
        # TODO: You can modify here according to the response format of the multi-turn model.
        response = json.loads(response)
        response = response.get("pre_sql", None)
        return response
    except Exception as e:
        logging.error(e)
        return None

