import tiktoken
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
import re


EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
FILEPATH = f"{os.getcwd()}/data"
FIGURE_INFO_DICT = {
    "agnes joaquim": {
        "formatted_name": "Agnes Joaquim",
        "category": "Cultural & Artistic Pioneers",
        "birth_year_iso": 1854,
        "birth_date_iso": "1854-04-07",
        "birth_date_display": "07 Apr 1854",
        "death_year_iso": 1899,
        "death_date_iso": "1899-07-02",
        "death_date_display": "2 Jul 1899",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=6afc206f-bb9c-44c4-9900-5130e65e90aa",
    },
    "checha davies": {
        "formatted_name": "Checha Davies",
        "category": "Social & Community Pioneers",
        "birth_year_iso": 1898,
        "birth_date_iso": "1898-01-01",
        "birth_date_display": "1898",
        "death_year_iso": 1979,
        "death_date_iso": "1979-09-02",
        "death_date_display": "02 Sep 1979",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=e93037c0-61d0-40bf-a8e6-d6ac843a84bb",
    },
    "david marshall": {
        "formatted_name": "David Marshall",
        "category": "Political Leaders & Nation Builders",
        "birth_year_iso": 1908,
        "birth_date_iso": "1908-03-12",
        "birth_date_display": "12 Mar 1908",
        "death_year_iso": 1995,
        "death_date_iso": "1995-12-12",
        "death_date_display": "12 Dec 1995",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=49e0d443-be0f-403c-93d5-1c9129ce16f0",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/david-marshall/story",
    },
    "elizabeth choy": {
        "formatted_name": "Elizabeth Choy",
        "category": "War Heroes & Resistance Fighters",
        "birth_year_iso": 1910,
        "birth_date_iso": "1910-11-29",
        "birth_date_display": "29 Nov 1910",
        "death_year_iso": 2006,
        "death_date_iso": "2006-09-14",
        "death_date_display": "14 Sep 2006",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=73f538cb-c39c-409d-b05e-f7c78480c606",
    },
    "g sarangapany": {
        "formatted_name": "G. Sarangapany",
        "category": "Cultural & Artistic Pioneers",
        "birth_year_iso": 1903,
        "birth_date_iso": "1903-04-20",
        "birth_date_display": "20 Apr 1903",
        "death_year_iso": 1974,
        "death_date_iso": "1974-03-16",
        "death_date_display": "16 Mar 1974",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=c6065a37-4f34-45dc-a8d1-7ddebe85de76",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/g-sarangapany/story",
    },
    "georgette chen": {
        "formatted_name": "Georgette Chen",
        "category": "Cultural & Artistic Pioneers",
        "birth_year_iso": 1906,
        "birth_date_iso": "1906-10-01",
        "birth_date_display": "Oct 1906",
        "death_year_iso": 1993,
        "death_date_iso": "1993-03-15",
        "death_date_display": "15 Mar 1993",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=0f964e7f-5382-416b-932b-3da4151f20e3",
    },
    "goh keng swee": {
        "formatted_name": "Goh Keng Swee",
        "category": "Political Leaders & Nation Builders",
        "birth_year_iso": 1918,
        "birth_date_iso": "1918-10-06",
        "birth_date_display": "6 Oct 1918",
        "death_year_iso": 2010,
        "death_date_iso": "2010-05-14",
        "death_date_display": "14 May 2010",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=73d1f784-af2b-402d-9fac-534e93db040d",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/goh-keng-swee/story",
    },
    "hajjah fatimah": {
        "formatted_name": "Hajjah Fatimah",
        "category": "Social & Community Pioneers",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=721b43a6-63d4-47cf-96ba-d47d0f9fe65d",
    },
    "lee kong chian": {
        "formatted_name": "Lee Kong Chian",
        "category": "Philanthropists & Business Pioneers",
        "birth_year_iso": 1893,
        "birth_date_iso": "1893-10-18",
        "birth_date_display": "18 Oct 1893",
        "death_year_iso": 1967,
        "death_date_iso": "1967-06-02",
        "death_date_display": "2 Jun 1967",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=513ccc41-7bb1-43fb-a9cc-bb8c2257db86",
    },
    "lee kuan yew": {
        "formatted_name": "Lee Kuan Yew",
        "category": "Political Leaders & Nation Builders",
        "birth_year_iso": 1923,
        "birth_date_iso": "1923-09-16",
        "birth_date_display": "16 Sep 1923",
        "death_year_iso": 2015,
        "death_date_iso": "2015-03-23",
        "death_date_display": "23 Mar 2015",
        "infopedia": {
            "01": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=5d7ebff1-0403-42bc-8f11-eee9f5967755",
            "02": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=84334fae-dc29-427b-9a5f-30b3d77e4087",
        },
        "roots": "https://www.roots.gov.sg/stories-landing/stories/lee-kuan-yew/story",
    },
    "lim bo seng": {
        "formatted_name": "Lim Bo Seng",
        "category": "War Heroes & Resistance Fighters",
        "birth_year_iso": 1909,
        "birth_date_iso": "1909-04-27",
        "birth_date_display": "27 Apr 1909",
        "death_year_iso": 1944,
        "death_date_iso": "1944-06-29",
        "death_date_display": "29 Jun 1944",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=18b892a1-9b32-4f8c-a72a-6ad72122bdd4",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/lim-bo-seng/story",
    },
    "lim kim san": {
        "formatted_name": "Lim Kim San",
        "category": "Political Leaders & Nation Builders",
        "birth_year_iso": 1916,
        "birth_date_iso": "1916-11-30",
        "birth_date_display": "30 Nov 1916",
        "death_year_iso": 2006,
        "death_date_iso": "2006-07-20",
        "death_date_display": "20 Jul 2006",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=0c994f60-ca57-4d67-8240-44eb77c5fd23",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/lim-kim-san/story",
    },
    "liu kang": {
        "formatted_name": "Liu Kang",
        "category": "Cultural & Artistic Pioneers",
        "birth_year_iso": 1911,
        "birth_date_iso": "1911-04-01",
        "birth_date_display": "1 Apr 1911",
        "death_year_iso": 2004,
        "death_date_iso": "2004-06-01",
        "death_date_display": "01 Jun 2004",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=5ffd9a55-8078-4df8-9652-ffc63a0ea2fe",
    },
    "munshi abdullah": {
        "formatted_name": "Munshi Abdullah",
        "category": "Cultural & Artistic Pioneers",
        "birth_year_iso": 1797,
        "birth_date_iso": "1797-01-01",
        "birth_date_display": "1797",
        "death_year_iso": 1854,
        "death_date_iso": "1854-10-01",
        "death_date_display": "1854",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=ab825e58-195e-4841-b873-b59d74494854",
    },
    "naraina pillai": {
        "formatted_name": "Naraina Pillai",
        "category": "Social & Community Pioneers",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=e751791b-cc41-4eb2-abb8-ff5b43b46ff9",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/naraina-pillai/story",
    },
    "ong teng cheong": {
        "formatted_name": "Ong Teng Cheong",
        "category": "Political Leaders & Nation Builders",
        "birth_year_iso": 1936,
        "birth_date_iso": "1936-01-22",
        "birth_date_display": "22 Jan 1936",
        "death_year_iso": 2002,
        "death_date_iso": "2002-02-08",
        "death_date_display": "8 Feb 2002",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=36ea0e62-7ab5-46e7-945a-9c2368f52fa7",
    },
    "percival frank aroozoo": {
        "formatted_name": "Percival Frank Aroozoo",
        "category": "Social & Community Pioneers",
        "birth_year_iso": 1900,
        "birth_date_iso": "1900-04-13",
        "birth_date_display": "13 Apr 1900",
        "death_year_iso": 1969,
        "death_date_iso": "1969-03-15",
        "death_date_display": "15 Mar 1969",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=625f30bc-7bb8-4fdf-8103-3b00f18eb41e",
    },
    "s rajaratnam": {
        "formatted_name": "S. Rajaratnam",
        "category": "Political Leaders & Nation Builders",
        "birth_year_iso": 1915,
        "birth_date_iso": "1915-02-25",
        "birth_date_display": "25 Feb 1915",
        "death_year_iso": 2006,
        "death_date_iso": "2006-02-22",
        "death_date_display": "22 Feb 2006",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=1a0dced9-8eae-486c-8320-140dd0ad15b3",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/sinnathamby-rajaratnam/story",
    },
    "stamford raffles": {
        "formatted_name": "Stamford Raffles",
        "category": "Colonial Figures",
        "birth_year_iso": 1781,
        "birth_date_iso": "1781-07-06",
        "birth_date_display": "6 Jul 1781",
        "death_year_iso": 1826,
        "death_date_iso": "1826-07-05",
        "death_date_display": "5 Jul 1826",
        "infopedia": {
            "01": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=1c9b75ae-8271-4a80-a272-ede763e2bd04",
            "02": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=3916c818-89dd-461b-9d45-81e27a08984a",
            "03": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=e3377388-98e6-412a-852a-6157970fc233",
            "04": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=64b03f60-837e-4565-b26d-d7c12d7ed8a4",
            "05": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=fcd18cc9-d547-45c2-805d-7ce3f1e84c1c",
            "06": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=8b54be13-5b70-44d0-abf6-f9781247686f",
        },
    },
    "syed omar bin ali aljunied": {
        "formatted_name": "Syed Omar bin Ali Aljunied",
        "category": "Social & Community Pioneers",
        "birth_year_iso": 1792,
        "birth_date_iso": "1792-01-01",
        "birth_date_display": "1792",
        "death_year_iso": 1852,
        "death_date_iso": "1852-11-06",
        "death_date_display": "6 Nov 1852",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=9473b3a1-7f27-45fa-b4f6-4e5f5bbb282d",
    },
    "tan kah kee": {
        "formatted_name": "Tan Kah Kee",
        "category": "Philanthropists & Business Pioneers",
        "birth_year_iso": 1874,
        "birth_date_iso": "1874-10-21",
        "birth_date_display": "21 Oct 1874",
        "death_year_iso": 1961,
        "death_date_iso": "1961-08-12",
        "death_date_display": "12 Aug 1961",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=41d208d1-4765-4b77-a1ff-4e4a0e3dfe2d",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/tan-kah-kee/story",
    },
    "tan tock seng": {
        "formatted_name": "Tan Tock Seng",
        "category": "Philanthropists & Business Pioneers",
        "birth_year_iso": 1798,
        "birth_date_iso": "1798-01-01",
        "birth_date_display": "1798",
        "death_year_iso": 1850,
        "death_date_iso": "1850-02-24",
        "death_date_display": "24 Feb 1850",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=35c1be2d-360b-4e36-90da-be55e5df0c69",
    },
    "william farquhar": {
        "formatted_name": "William Farquhar",
        "category": "Colonial Figures",
        "birth_year_iso": 1774,
        "birth_date_iso": "1774-02-26",
        "birth_date_display": "26 Feb 1774",
        "death_year_iso": 1839,
        "death_date_iso": "1839-05-11",
        "death_date_display": "11 May 1839",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=0341bd61-8f21-47b8-a774-de9f72bcfa04",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/the-first-resident/the-first-resident",
    },
    "yusof bin ishak": {
        "formatted_name": "Yusof bin Ishak",
        "category": "Political Leaders & Nation Builders",
        "birth_year_iso": 1910,
        "birth_date_iso": "1910-08-12",
        "birth_date_display": "12 Aug 1910",
        "death_year_iso": 1970,
        "death_date_iso": "1970-11-23",
        "death_date_display": "23 Nov 1970",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=71b61553-4d2b-4f7d-80c6-71749619d172",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/yusof-ishak/story",
    },
    "zubir said": {
        "formatted_name": "Zubir Said",
        "category": "Cultural & Artistic Pioneers",
        "birth_year_iso": 1907,
        "birth_date_iso": "1907-07-22",
        "birth_date_display": "22 Jul 1907",
        "death_year_iso": 1987,
        "death_date_iso": "1987-11-16",
        "death_date_display": "16 Nov 1987",
        "infopedia": "www.nlb.gov.sg/main/article-detail?cmsuuid=4b3061ce-c763-480d-9867-ac364bb139bf",
        "roots": "https://www.roots.gov.sg/stories-landing/stories/zubir-said/story",
    },
}

ICH_INFO_DICT = {
    "cheongsam tailoring": {
        "formatted_name": "Cheongsam Tailoring",
        "categories": [
            "Traditional Craftsmanship",
            "Social Practices, Rituals and Festive Events",
        ],
        "ref_no": "ICH-042",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/cheongsam-tailoring",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=9893f3e4-b524-4f12-ae63-5581a712be84",
    },
    "chicken rice": {
        "formatted_name": "Chicken Rice",
        "categories": ["Food Heritage"],
        "ref_no": "ICH-064",
        "date_of_inclusion": "March 2019",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=ceddd346-4072-4981-b20d-f771bea7dd81",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/chicken-rice",
    },
    "chinese opera": {
        "formatted_name": "Chinese Opera",
        "categories": ["Performing Arts"],
        "ref_no": "ICH-006",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "https://www.roots.gov.sg/ich- landing/ich/chinese-opera",
    },
    "chingay": {
        "formatted_name": "Chingay",
        "categories": ["Social Practices, Rituals and Festive Events"],
        "ref_no": "ICH-098",
        "date_of_inclusion": "March 2022",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=c5b0f751-ad72-47d1-ad4a-8ff51d0607e4",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/chingay",
    },
    "craft and practices related to kebaya": {
        "formatted_name": "Craft and Practices Related to Kebaya",
        "categories": [
            "Traditional Craftsmanship",
            "Social Practices, Rituals and Festive Events",
        ],
        "ref_no": "ICH-102",
        "date_of_inclusion": "October 2022; Updated March 2023",
        "roots": "https://www.roots.gov.sg/ich- landing/ich/Kebaya",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=4c73810c-81ff-4ec3-822c-584a40d9d84d",
    },
    "dikir barat": {
        "formatted_name": "Dikir Barat",
        "categories": ["Oral Traditions and Expressions", "Performing Arts"],
        "ref_no": "ICH-004",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "https://www.roots.gov.sg/ich- landing/ich/dikir-barat",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=2507d62a-542e-4878-a18b-6997c04b3381",
    },
    "eurasian cuisine in singapore": {
        "formatted_name": "Eurasian Cuisine in Singapore",
        "categories": ["Food Heritage"],
        "ref_no": "ICH-049",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "www.roots.gov.sg/ich-landing/ich/eurasian-cuisine-in-singapore",
    },
    "getai": {
        "formatted_name": "Getai",
        "categories": [
            "Performing Arts",
            "Social Practices, Rituals and Festive Events",
            "Oral Traditions and Expressions",
        ],
        "ref_no": "ICH-057",
        "date_of_inclusion": "March 2019",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=5f641690-731c-4816-b84b-733232c8d44f",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/getai",
    },
    "hainanese curry rice": {
        "formatted_name": "Hainanese Curry Rice",
        "categories": ["Food Heritage"],
        "ref_no": "ICH-061",
        "date_of_inclusion": "March 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/hainanese-curry-rice",
    },
    "hari raya puasa": {
        "formatted_name": "Hari Raya Puasa",
        "categories": ["Social Practices, Rituals and Festive Events"],
        "ref_no": "ICH-022",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/hari-raya-puasa",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=1d986272-ab7e-4e5c-af7c-5b69e76c60af",
    },
    "hawker culture": {
        "formatted_name": "Hawker Culture",
        "categories": [
            "Food Heritage",
            "Social Practices, Rituals and Festive Events",
            "Traditional Craftsmanship",
        ],
        "ref_no": "ICH-050",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/hawker-culture",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=d30b9e62-9e71-4e1e-9f5e-a6573380f4b6",
    },
    "indian dance forms": {
        "formatted_name": "Indian Dance Forms",
        "categories": ["Performing Arts"],
        "ref_no": "ICH-011",
        "date_of_inclusion": "April 2018; Updated March 2025",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/indian-dance-forms",
    },
    "kueh": {
        "formatted_name": "Kueh",
        "categories": ["Food Heritage"],
        "ref_no": "ICH-085",
        "date_of_inclusion": "October 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/kueh",
    },
    "lion dance": {
        "formatted_name": "Lion Dance",
        "categories": ["Social Practices, Rituals and Festive Events"],
        "ref_no": "ICH-016",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "https://www.roots.gov.sg/ich- landing/ich/lion-dance",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=f5cf51ef-34fd-45ac-833f-3503bbc108a4",
    },
    "making and use of sambal": {
        "formatted_name": "Making and Use of Sambal",
        "categories": ["Food Heritage"],
        "ref_no": "ICH-99",
        "date_of_inclusion": "March 2022",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/Making-and-Use-of-Sambal",
    },
    "nyonya beadwork and embroidery": {
        "formatted_name": "Nyonya Beadwork and Embroidery",
        "categories": [
            "Social Practices, Rituals and Festive Events",
            "Traditional Craftsmanship",
        ],
        "ref_no": "ICH-044",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/nyonya-beadwork-and-embroidery",
    },
    "orchid cultivation": {
        "formatted_name": "Orchid Cultivation",
        "categories": [
            "Social Practices, Rituals and Festive Events",
            "Knowledge and Practices concerning Nature and Universe",
        ],
        "ref_no": "ICH-079",
        "date_of_inclusion": "October 2019; Updated March 2025",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/orchid-cultivation",
    },
    "thaipusam": {
        "formatted_name": "Thaipusam",
        "categories": ["Social Practices, Rituals and Festive Events"],
        "ref_no": "ICH-027",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=deea4c72-513b-4f08-95a7-3f41b5a0eff0",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/thaipusam",
    },
    "the jinkli nona song and the branyo dance": {
        "formatted_name": "The Jinkli Nona Song and The Branyo Dance",
        "categories": ["Oral Traditions and Expressions", "Performing Arts"],
        "ref_no": "ICH-053",
        "date_of_inclusion": "March 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/The-jinkli-nona-song-and-The-branyo-dance",
    },
    "traditional malay music": {
        "formatted_name": "Traditional Malay Music",
        "categories": ["Performing Arts"],
        "ref_no": "ICH-071",
        "date_of_inclusion": "October 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/traditional-malay-music",
    },
    "vesak day": {
        "formatted_name": "Vesak Day",
        "categories": ["Social Practices, Rituals and Festive Events"],
        "ref_no": "ICH-031",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=66053540-f81f-47c8-82ce-6095d5b599c3",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/vesak-day",
    },
    "wayang kulit": {
        "formatted_name": "Wayang Kulit",
        "categories": ["Oral Traditions and Expressions", "Performing Arts"],
        "ref_no": "ICH-009",
        "date_of_inclusion": "April 2018; Updated March 2019",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=ac6436e3-6eef-41ad-a5b9-39400606896b",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/wayang-kulit",
    },
    "wayang peranakan": {
        "formatted_name": "Wayang Peranakan",
        "categories": ["Oral Traditions and Expressions", "Performing Arts"],
        "ref_no": "ICH-051",
        "date_of_inclusion": "March 2019",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/wayang-peranakan",
    },
    "weaving ketupat": {
        "formatted_name": "Weaving Ketupat",
        "categories": [
            "Social Practices, Rituals and Festive Events",
            "Traditional Craftsmanship",
            "Food Heritage",
        ],
        "ref_no": "ICH-076",
        "date_of_inclusion": "October 2019; Updated March 2023",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=cf26883a-d1ab-426c-908a-322fb5561a0c",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/weaving-ketupat",
    },
    "rangoli": {
        "formatted_name": "Rangoli",
        "categories": [
            "Social Practices, Rituals and Festive Events",
            "Knowledge and Practices concerning Nature and Universe",
            "Traditional Craftsmanship",
        ],
        "ref_no": "ICH-058",
        "date_of_inclusion": "March 2019",
        "infopedia": "https://www.nlb.gov.sg/main/article-detail?cmsuuid=daeab56f-2261-4fb5-ac9b-7752449d1854",
        "roots": "https://www.roots.gov.sg/ich-landing/ich/rangoli",
    },
}


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))


def get_vectordb(load_documents: bool = False, is_figures: bool = True):
    if load_documents:
        headers_to_split_on = [("#", "Title"), ("##", "Section"), ("###", "Subsection")]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
        )

        docs_prepared = []

        for dirpath, _, filenames in os.walk(FILEPATH):
            folder = "person" if is_figures else "ich"
            for filename in [f for f in filenames if folder in dirpath]:
                try:
                    source = filename.split("_")[0]
                    entity_name = (
                        " ".join(
                            re.sub(
                                r"_\d+",
                                "",
                                filename.replace(".md", "").replace(source, "_"),
                            ).split("_")
                        )
                        .strip()
                        .lower()
                    )

                    text_file_path = os.path.join(dirpath, filename)

                    with open(text_file_path, "r", encoding="utf-8") as f:
                        data = f.read()
                    source = (
                        "Singapore Infopedia" if source == "infopedia" else "Roots.sg"
                    )

                    markdown_chunks = markdown_splitter.split_text(data)
                    final_chunks = recursive_splitter.split_documents(markdown_chunks)

                    print(f"{entity_name}: {len(final_chunks)}")
                    for chunk in final_chunks:
                        if is_figures:
                            figure_details = FIGURE_INFO_DICT[entity_name]

                            metadata_for_figure = {
                                "category": figure_details.get("category", None),
                                "birth_year_iso": figure_details.get(
                                    "birth_year_iso", None
                                ),
                                "birth_date_iso": figure_details.get(
                                    "birth_date_iso", None
                                ),
                                "birth_date_display": figure_details.get(
                                    "birth_date_display", None
                                ),
                                "death_year_iso": figure_details.get(
                                    "death_year_iso", None
                                ),
                                "death_date_iso": figure_details.get(
                                    "death_date_iso", None
                                ),
                                "death_date_display": figure_details.get(
                                    "death_date_display", None
                                ),
                            }

                            article_details = (
                                figure_details["infopedia"]
                                if source == "Singapore Infopedia"
                                else figure_details["roots"]
                            )

                            if isinstance(article_details, str):
                                article_link = article_details
                            else:
                                for article_num in article_details.keys():
                                    if article_num in text_file_path:
                                        article_link = article_details[article_num]
                                        break

                            metadata_for_figure["article_url"] = article_link

                            metadata_for_figure = {
                                key: value
                                for key, value in metadata_for_figure.items()
                                if value
                            }

                            docs_prepared.append(
                                Document(
                                    page_content=f"[Person: {entity_name}, Category: {metadata_for_figure['category']}, Source: {source}]\n\n{chunk.page_content}",
                                    metadata={
                                        **chunk.metadata,
                                        "source": source,
                                        "person": entity_name,
                                        **metadata_for_figure,
                                    },
                                )
                            )
                        else:
                            if entity_name == "ich":
                                docs_prepared.append(
                                    Document(
                                        page_content=f"[Info: Background of ICH, Source: {source}]\n\n{chunk.page_content}",
                                        metadata={
                                            **chunk.metadata,
                                            "source": source,
                                            "info": "Background of ICH",
                                            "article_url": "https://www.roots.gov.sg/ich%20landing/about-intangible-cultural-heritage",
                                        },
                                    )
                                )
                            else:
                                element_details = ICH_INFO_DICT[entity_name]
                                metadata_for_element = {
                                    "categories": (
                                        ", ".join(element_details["categories"])
                                        if element_details["categories"]
                                        else None
                                    ),
                                    "ref_no": element_details.get("ref_no", None),
                                    "date_of_inclusion": element_details.get(
                                        "date_of_inclusion", None
                                    ),
                                    "article_url": (
                                        element_details.get("infopedia", None)
                                        if source == "Singapore Infopedia"
                                        else element_details.get("roots", None)
                                    ),
                                }

                                docs_prepared.append(
                                    Document(
                                        page_content=f"[Element: {entity_name}, Ref No.: {metadata_for_element['ref_no']}, Categories: {metadata_for_element['categories']}, Source: {source}]\n\n{chunk.page_content}",
                                        metadata={
                                            **chunk.metadata,
                                            "source": source,
                                            "element": entity_name,
                                            **metadata_for_element,
                                        },
                                    )
                                )
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue

        vectordb = Chroma.from_documents(
            documents=docs_prepared,
            collection_name=(
                "singaroots_historical_figures" if is_figures else "singaroots_ich"
            ),
            persist_directory="./singaroots_vector_db",
            embedding=EMBEDDINGS_MODEL,
        )
    else:
        vectordb = Chroma(
            collection_name=(
                "singaroots_historical_figures" if is_figures else "singaroots_ich"
            ),
            persist_directory="./singaroots_vector_db",
            embedding_function=EMBEDDINGS_MODEL,
        )

    return vectordb
