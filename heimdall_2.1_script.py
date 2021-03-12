# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:33:01 2021

@author: ztche
"""

#import Scrapy, pandas & datetime
import scrapy
import pandas as pd
from scrapy.crawler import CrawlerProcess
from datetime import datetime
from datetime import date

#body of spider
class MySpider(scrapy.Spider):
    name = "Heimdall"
    
    def start_requests(self):
        
        #search term input
        terms = [term for term in input("Search Terms (space-seperated): ").split()] 
        print("Heimdall Conversion: " + str(terms))
        for term in terms:
            urls = ['https://s.weibo.com/weibo?q=' + term + '&Refer=SWeibo_box']
            for url in urls:
                yield scrapy.Request(url = url, callback = self.parse)
                
    def parse(self, response):

        #selectors
        posts = response.css( '.content' )
        author = posts.css( '.name' )
        text = posts.xpath( './/p[@node-type="feed_list_content"]' )
        info = posts.css( '.from' )
        search_term = response.css( 'div.search-input ::attr(value)' ).get()

        #counter
        x = -1

        #display search info
        print("\nSearch Term: " + str(search_term))
        print("Results: " + str(len(text)))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Search Completed: " + str(date.today()) + " at " + str(current_time))

        #select search contents
        author_stripped = []
        text_stripped = []
        info_stripped = []
        for t in text:
            x += 1
            counter_author = author[x]
            counter_content = text[x] 
            counter_info = info[x]
            author_extracts = counter_author.css( '::text' ).getall()
            text_extracts = counter_content.css( '::text' ).getall()
            info_extracts = counter_info.css( '::text' ).getall()
            author_stripped.append([a.strip() for a in author_extracts])
            text_stripped.append([b.strip() for b in text_extracts])
            info_stripped.append([c.strip() for c in info_extracts])
        
        #visualise using DataFrame
        data = {'Author' : author_stripped,
                'Content' : text_stripped,
                'Info' : info_stripped
                }
        df = pd.DataFrame(data, columns = ['Author', 'Content', 'Info'])
        pd.set_option('display.max_colwidth', None)
        display(df)

#run spider
process = CrawlerProcess()
process.crawl(MySpider)
process.start()