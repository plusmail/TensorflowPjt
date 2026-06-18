#-*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import urlparse
from PIL import Image
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import time
import os

count1 = 0

fileDir = './images'
catDir = fileDir + '/cat/'
dogDir = fileDir + '/dog_v1/'
conversionDir = fileDir + '/conversionDir/'

def getImage(page, Word):
    start = time.time()
    base_url = 'https://www.google.co.kr'
    img_list = []

    opts = Options()
    opts.add_argument("user-agent=whatever you want")

    browser = webdriver.Chrome(options=opts)

    browser.get(base_url)

    assert "Google" in browser.title

    elem = browser.find_element(By.NAME, "q")
    elem.clear()

    elem.send_keys(Word)
    elem.submit()

    elem = browser.find_element(By.XPATH, '//*[@id="mn"]/tbody[1]/tr[3]/td[2]/div[1]/div[2]').click()
    elem = browser.find_element(By.XPATH, '//*[@id="ires"]/table/tbody')
    elem = browser.find_element(By.CSS_SELECTOR, '#isz_l > a').click()

    count = 1
    b = 0
    while count <= page:
        print('\n-----', count, '-----')

        html = browser.page_source
        soup = BeautifulSoup(html, 'html5lib')
        img = soup.find_all("img")

        for a in img:
            img_list.append(a.get('src'))
            o = urlparse(a.get('src'))
            img_name = img_list[b].replace("/", "")

            if o.scheme:
                if not duplicate(img_name):
                    urllib.request.urlretrieve(a.get('src'), dogDir + str(img_name))
                    print(str(b) + ": " + str(img_name))
                else:
                    print("Duplicate image!!!!!")
            b += 1

        elem = browser.find_element(By.CSS_SELECTOR, '#nav > tbody > tr > td:nth-child(12)').click()
        count += 1

    print("크롤링 소요 시간:", round(time.time() - start, 6))
    assert "No results found." not in browser.page_source
    browser.close()

def imgResize():
    imagefiles = [os.path.join(dogDir, fileName) for fileName in os.listdir(dogDir)]
    for fileName in imagefiles:
        if "https" in fileName:
            img = Image.open(fileName)
            reimg = img.resize((250, 250), resample=Image.LANCZOS)
            reimg.save(conversionDir + str(img) + '.JPEG', format='JPEG', compress_level=10, quality=100)
        else:
            print("Not resize")
    print("이미지 리사이즈 종료-----")

def duplicate(img):
    return os.path.exists(dogDir + img)

if __name__ == "__main__":
    num = int(input("페이지: "))
    searchWord = str(input("검색어: "))
    getImage(num, searchWord)
    imgResize()