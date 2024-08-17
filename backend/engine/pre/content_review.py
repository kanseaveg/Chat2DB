"""
   Content review: We use Baidu's content review API.
"""
import logging


async def content_review(state, text):
    baidu_client = state.baidu_client
    try:
        result = baidu_client.textCensorUserDefined(text)
        if result['conclusionType'] == 1:
            logging.info("current content is legal")
            return True
        else:
            logging.info("current content is illegal. please adjust the content.")
            return False
    except Exception as e:
        logging.error("current baidu content review connected error: %s" % e)
        return False




