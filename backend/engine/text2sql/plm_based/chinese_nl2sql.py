"""
  Optional
"""
import json
import logging
async def single_turn_nl2sql(state, question, db):
    settings = state.settings
    client = state.client
    url = settings.CHINESE_SINGLE_TURN_SERVER_ADDRESS
    try:
        response = await client.post(
            url,
            json={
                "question": question,
                "db": db,
            }
        )
        # TODO: You can modify here according to the response format of the multi-turn model.
        response = json.loads(response)
        response = response.get("pre_sql", None)
        return response
    except Exception as e:
        logging.error(e)
        return None