# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:18:04 2021

@author: ztche
"""

# imports
import scrapy
import pandas as pd
from scrapy.crawler import CrawlerProcess
from datetime import datetime
from datetime import date
import numpy as np

# functions
def clean(list):
    """returns .getall() list as uninterrupted text"""
    result= ""
    remove = ["\n"]
    for e in list:
        for r in remove:
            if r in e:
                e = e.replace(r, "")
        result += str(e)                
    yield result.strip()

# body of spider
class MySpider(scrapy.Spider):
    name = "Heimdall"
    
    def start_requests(self):
        
        # search term input
        terms = [term for term in input("Search Terms (space-seperated): ").split()]
        for term in terms:
            urls = ['https://s.weibo.com/weibo?q=' + term + '&Refer=SWeibo_box']
            for url in urls:
                yield scrapy.Request(url = url, callback = self.parse)
                
    def parse(self, response):

        # selectors
        posts = response.css( '.content' )
        author = posts.css( '.name' )
        text = posts.xpath( './/p[@node-type="feed_list_content"]' )
        info = posts.css( '.from' )
        search_term = response.css( 'div.search-input ::attr(value)' ).get()

        # counter
        x = -1

        # display search info
        print("\nSearch Term: " + str(search_term))
        print("Results: " + str(len(text)))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Search Completed: " + str(date.today()) + " at " + str(current_time))

        # select search contents
        author_data = []
        text_data = []
        info_data = []
        for t in text:
            x += 1
            counter_author = author[x]
            counter_content = text[x] 
            counter_info = info[x]
            author_extracts = counter_author.css( '::text' ).getall()
            text_extracts = counter_content.css( '::text' ).getall()
            info_extracts = counter_info.css( '::text' ).getall()
            
            # append data
            author_data.append([a for a in clean(author_extracts)])
            text_data.append([t for t in clean(text_extracts)])
            info_data.append([i for i in clean(info_extracts)])
        
        # visualise using DataFrame
        data = {'Author' : author_data,
                'Content' : text_data,
                'Info' : info_data
                }
        df = pd.DataFrame(data, columns = ['Author', 'Content', 'Info'])
        df.index = np.arange(1, len(df)+1)
        pd.set_option('display.max_colwidth', None)
        display(df)

# run spider
process = CrawlerProcess()
process.crawl(MySpider)
process.start()