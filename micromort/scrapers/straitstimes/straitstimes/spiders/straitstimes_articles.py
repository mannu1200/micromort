# -*- coding: utf-8 -*-
import scrapy
from micromort.resources.configs.mongodbconfig import mongodb_config
from micromort.scrapers.straitstimes.straitstimes.items import StraitsTimesArticlesItem
import pymongo
import logging
import os
from scrapy.utils.log import configure_logging

import micromort.constants as constants
PATHS = constants.PATHS
LOGS_DIR = os.path.join(PATHS['LOGS_DIR'], 'straitstimes_spider')


class StraitstimesNewsArticles(scrapy.Spider):
    name = 'straitstimes_articles'
    allowed_domains = ['straitstimes.com']
    straitstimes_base_url = 'http://straitstimes.com'
    custom_settings = {
        'ITEM_PIPELINES': {
            'micromort.scrapers.straitstimes.straitstimes.pipelines.DuplicateNewsPipeline': 100,
            'micromort.scrapers.straitstimes.straitstimes.pipelines.MongoDBPipeline': 200,
        }
    }

    configure_logging(settings={
        'LOG_LEVEL': 'INFO'
    })
    logging.basicConfig(
        filename=os.path.join(LOGS_DIR, 'log.txt'),
        level=logging.INFO,
        filemode='a'
    )

    def __init__(self, *args, **kwargs):
        self.MONGODB_URL = mongodb_config['host']
        self.MONGODB_PORT = mongodb_config['port']
        self.MONGODB_DB = mongodb_config['db']
        self.ARTICLE_COLLECTION = mongodb_config['straitstimes_article_collection']
        self.HEADLINE_COLLECTION = mongodb_config['straitstimes_headlines_collection']
        self.client = pymongo.MongoClient(self.MONGODB_URL, self.MONGODB_PORT)
        self.db = self.client[self.MONGODB_DB]
        self.collection = self.db[self.ARTICLE_COLLECTION]
        self.headlines_collection = self.db[self.HEADLINE_COLLECTION]
        self.article_urls = self.headlines_collection.distinct('article_url')
        self.unique_index = 'article_url'  # used to check the duplicates in the collection
        super(StraitstimesNewsArticles, self).__init__(*args, **kwargs)

    def parse(self, response):
        article_div = response.css('div[itemprop="articleBody"]')
        if len(article_div) > 0:
            article_text = article_div[0].xpath('p/text()').extract()
            article_text = ''.join(article_text)
            url = response.url
            item = StraitsTimesArticlesItem(
                article_url=url,
                article_text=article_text
            )
            yield item
        else:
            self.logger.info("URL {0} is not parsed because articleBody is not found \n".format(response.url))

    def start_requests(self):
        """
        This will generate the urls from mongodb to scrape the main article page
        """
        for url in self.article_urls:
            url = self.straitstimes_base_url + '/' + url
            yield scrapy.http.Request(url, callback=self.parse)


