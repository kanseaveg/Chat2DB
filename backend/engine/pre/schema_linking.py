import logging
logging.basicConfig(level=logging.INFO)
import json
async def schema_linking(state, question, db, selected_schemas):
    settings = state.settings
    client = state.client
    url = settings.SCHEMA_LINKING_SERVICE_ADDRESS
    print(url)
    try:
        response = await client.post(
            url,
            json={
                "question": question,
                "db": db,
                "selected_schemas": selected_schemas
            }
        )
        # TODO: You can modify here according to the response format of the multi-turn model.
        response = json.loads(response)
        response = response.get("q_s", None)
        # logging.info(f"q_s is {response}")
        return response
    except Exception as e:
        logging.error(e)
        return None