#!/usr/bin/env python3
"""
Universal Museum Search Script
Provides unified interface for searching across multiple museum collections
"""

import requests
import json
from urllib.parse import urlencode, quote
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SearchParams:
    """Unified search parameters across all museums"""

    keywords: Optional[str] = None
    location: Optional[str] = None  # Geographic origin/place
    date_start: Optional[int] = None  # Year
    date_end: Optional[int] = None  # Year
    era: Optional[str] = None  # 'ad', 'bc', 'ce', 'bce'
    has_image: bool = True
    page: int = 1
    limit: int = 20  # Results per page
    sort_by: Optional[str] = None  # 'relevance', 'date', 'title', 'artist'
    sort_order: str = "asc"  # 'asc', 'desc'


class MuseumSearcher(ABC):
    """Base class for museum searchers"""

    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key

    @abstractmethod
    def search(self, params: SearchParams) -> Dict:
        """Execute search and return results"""
        pass

    @abstractmethod
    def get_search_url(self, params: SearchParams) -> str:
        """Generate the search URL for this museum"""
        pass


class BritishMuseumSearcher(MuseumSearcher):
    """British Museum collection search"""

    def __init__(self):
        super().__init__("British Museum")
        self.base_url = "https://www.britishmuseum.org/collection"

    def get_search_url(self, params: SearchParams) -> str:
        query_params = {}

        if params.keywords:
            query_params["keyword"] = params.keywords
        if params.location:
            query_params["place"] = params.location
        if params.has_image:
            query_params["image"] = "true"
        if params.date_end:
            query_params["dateTo"] = str(params.date_end)
        if params.date_start:
            query_params["dateFrom"] = str(params.date_start)
        if params.era:
            query_params["eraTo"] = params.era
        if params.page:
            query_params["page"] = str(params.page)
        if params.sort_by == "date":
            query_params["sort"] = (
                "date__asc" if params.sort_order == "asc" else "date__desc"
            )
        elif params.sort_by == "location":
            query_params["sort"] = "production_place__asc"

        query_params["view"] = "grid"

        return f"{self.base_url}/search?" + urlencode(query_params)

    def search(self, params: SearchParams) -> Dict:
        url = self.get_search_url(params)
        try:
            response = requests.get(url)
            return {
                "museum": self.name,
                "url": url,
                "status": response.status_code,
                "success": response.status_code == 200,
                "note": "Web scraping required for results parsing",
            }
        except Exception as e:
            return {"museum": self.name, "url": url, "error": str(e), "success": False}


class UPennMuseumSearcher(MuseumSearcher):
    """UPenn Museum collection search"""

    def __init__(self):
        super().__init__("UPenn Museum")
        self.base_url = "https://www.penn.museum/collections/search.php"

    def get_search_url(self, params: SearchParams) -> str:
        query_params = {}

        if params.keywords:
            query_params["term"] = params.keywords
        if params.has_image:
            query_params["images[]"] = "yes"

        query_params["submit_term"] = "Submit Query"

        return f"{self.base_url}?" + urlencode(query_params, doseq=True)

    def search(self, params: SearchParams) -> Dict:
        url = self.get_search_url(params)
        try:
            response = requests.get(url)
            return {
                "museum": self.name,
                "url": url,
                "status": response.status_code,
                "success": response.status_code == 200,
                "note": "Web scraping required for results parsing",
            }
        except Exception as e:
            return {"museum": self.name, "url": url, "error": str(e), "success": False}


class MFABostonSearcher(MuseumSearcher):
    """Boston Museum of Fine Arts collection search"""

    def __init__(self):
        super().__init__("Boston MFA")
        self.base_url = "https://collections.mfa.org/search/Objects"

    def get_search_url(self, params: SearchParams) -> str:
        # MFA uses path-based search terms
        search_path = f"*/{quote(params.keywords or 'art')}"
        query_params = {}

        if params.has_image:
            query_params["filter"] = "imageExistence:true"
        if params.page and params.page > 1:
            query_params["page"] = str(params.page)
        if params.sort_by == "title":
            query_params["sort"] = (
                "title-asc" if params.sort_order == "asc" else "title-desc"
            )
        elif params.sort_by == "date":
            query_params["sort"] = (
                "displayDate-asc" if params.sort_order == "asc" else "displayDate-desc"
            )

        url = f"{self.base_url}/{search_path}"
        if query_params:
            url += "?" + urlencode(query_params)

        return url

    def search(self, params: SearchParams) -> Dict:
        url = self.get_search_url(params)
        try:
            response = requests.get(url)
            return {
                "museum": self.name,
                "url": url,
                "status": response.status_code,
                "success": response.status_code == 200,
                "note": "Web scraping required for results parsing",
            }
        except Exception as e:
            return {"museum": self.name, "url": url, "error": str(e), "success": False}


class MoMASearcher(MuseumSearcher):
    """Museum of Modern Art API search"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("MoMA", api_key)
        self.base_url = "https://api.moma.org"
        self.web_url = "https://www.moma.org/collection"

    def get_search_url(self, params: SearchParams) -> str:
        if self.api_key:
            # Use API endpoint
            query_params = {"api_key": self.api_key}
            if params.keywords:
                query_params["q"] = params.keywords
            if params.has_image:
                query_params["has_image"] = "true"
            if params.page:
                query_params["page"] = str(params.page)
            if params.date_start:
                query_params["date_begin"] = str(params.date_start)
            if params.date_end:
                query_params["date_end"] = str(params.date_end)

            return f"{self.base_url}/collection/works/search?" + urlencode(query_params)
        else:
            # Use web interface
            query_params = {}
            if params.keywords:
                query_params["q"] = params.keywords
            if params.has_image:
                query_params["has_image"] = "true"
            if params.page:
                query_params["page"] = str(params.page)

            return f"{self.web_url}/works/search?" + urlencode(query_params)

    def search(self, params: SearchParams) -> Dict:
        url = self.get_search_url(params)
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(url, headers=headers)

            result = {
                "museum": self.name,
                "url": url,
                "status": response.status_code,
                "success": response.status_code == 200,
            }

            if response.status_code == 200 and self.api_key:
                try:
                    data = response.json()
                    result["data"] = data
                    result["count"] = data.get("count", 0)
                except:
                    result["note"] = "Response not JSON parseable"
            else:
                result["note"] = (
                    "Web scraping required" if not self.api_key else "API error"
                )

            return result
        except Exception as e:
            return {"museum": self.name, "url": url, "error": str(e), "success": False}


class SmithsonianSearcher(MuseumSearcher):
    """Smithsonian Open Access API search"""

    def __init__(self, api_key: str):
        super().__init__("Smithsonian", api_key)
        self.base_url = "https://api.si.edu/openaccess/api/v1.0"

    def get_search_url(self, params: SearchParams) -> str:
        query_params = {"api_key": self.api_key}

        # Build search query
        search_terms = []
        if params.keywords:
            search_terms.append(params.keywords)
        if params.location:
            search_terms.append(f'place:"{params.location}"')

        if search_terms:
            query_params["q"] = " AND ".join(search_terms)

        if params.has_image:
            query_params["fq"] = "online_media_type:Images"
        if params.limit:
            query_params["rows"] = str(min(params.limit, 100))  # Max 100
        if params.page and params.page > 1:
            query_params["start"] = str((params.page - 1) * params.limit)

        return f"{self.base_url}/search?" + urlencode(query_params)

    def search(self, params: SearchParams) -> Dict:
        url = self.get_search_url(params)
        try:
            response = requests.get(url)

            result = {
                "museum": self.name,
                "url": url,
                "status": response.status_code,
                "success": response.status_code == 200,
            }

            if response.status_code == 200:
                try:
                    data = response.json()
                    result["data"] = data
                    result["count"] = data.get("response", {}).get("numFound", 0)
                    result["objects"] = data.get("response", {}).get("docs", [])
                except:
                    result["note"] = "Response not JSON parseable"

            return result
        except Exception as e:
            return {"museum": self.name, "url": url, "error": str(e), "success": False}


class HarvardSearcher(MuseumSearcher):
    """Harvard Art Museums API search"""

    def __init__(self, api_key: str):
        super().__init__("Harvard Art Museums", api_key)
        self.base_url = "https://api.harvardartmuseums.org"

    def get_search_url(self, params: SearchParams) -> str:
        query_params = {"apikey": self.api_key}

        if params.keywords:
            query_params["q"] = params.keywords
        if params.has_image:
            query_params["hasimage"] = "1"
        if params.limit:
            query_params["size"] = str(min(params.limit, 100))  # Max 100
        if params.page:
            query_params["page"] = str(params.page)
        if params.location:
            query_params["culture"] = params.location
        if params.sort_by == "title":
            query_params["sort"] = "title"
            query_params["sortorder"] = params.sort_order
        elif params.sort_by == "date":
            query_params["sort"] = "dated"
            query_params["sortorder"] = params.sort_order

        return f"{self.base_url}/object?" + urlencode(query_params)

    def search(self, params: SearchParams) -> Dict:
        url = self.get_search_url(params)
        try:
            response = requests.get(url)

            result = {
                "museum": self.name,
                "url": url,
                "status": response.status_code,
                "success": response.status_code == 200,
            }

            if response.status_code == 200:
                try:
                    data = response.json()
                    result["data"] = data
                    result["count"] = data.get("info", {}).get("totalrecords", 0)
                    result["objects"] = data.get("records", [])
                except:
                    result["note"] = "Response not JSON parseable"

            return result
        except Exception as e:
            return {"museum": self.name, "url": url, "error": str(e), "success": False}


class UniversalMuseumSearcher:
    """Main class that coordinates searches across all museums"""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize with API keys for museums that require them
        api_keys format: {
            'smithsonian': 'your_smithsonian_key',
            'harvard': 'your_harvard_key',
            'moma': 'your_moma_key'  # optional
        }
        """
        api_keys = api_keys or {}

        self.searchers = {
            "british_museum": BritishMuseumSearcher(),
            "upenn_museum": UPennMuseumSearcher(),
            "mfa_boston": MFABostonSearcher(),
            "moma": MoMASearcher(api_keys.get("moma")),
        }

        # Add API-based museums if keys provided
        if "smithsonian" in api_keys:
            self.searchers["smithsonian"] = SmithsonianSearcher(api_keys["smithsonian"])

        if "harvard" in api_keys:
            self.searchers["harvard"] = HarvardSearcher(api_keys["harvard"])

    def search_all(
        self, params: SearchParams, museums: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """Search all or specified museums with unified parameters"""
        museums_to_search = museums or list(self.searchers.keys())
        results = {}

        for museum_key in museums_to_search:
            if museum_key in self.searchers:
                try:
                    results[museum_key] = self.searchers[museum_key].search(params)
                except Exception as e:
                    results[museum_key] = {
                        "museum": self.searchers[museum_key].name,
                        "error": str(e),
                        "success": False,
                    }

        return results

    def get_all_urls(
        self, params: SearchParams, museums: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Get search URLs for all museums without executing requests"""
        museums_to_search = museums or list(self.searchers.keys())
        urls = {}

        for museum_key in museums_to_search:
            if museum_key in self.searchers:
                try:
                    urls[museum_key] = self.searchers[museum_key].get_search_url(params)
                except Exception as e:
                    urls[museum_key] = f"Error generating URL: {e}"

        return urls


def main():
    """Example usage of the Universal Museum Searcher"""

    # API keys (replace with your actual keys)
    api_keys = {
        # 'smithsonian': 'your_smithsonian_api_key_here',
        # 'harvard': 'your_harvard_api_key_here',
        # 'moma': 'your_moma_api_key_here'  # Optional
    }

    # Initialize searcher
    searcher = UniversalMuseumSearcher(api_keys)

    # Define search parameters
    search_params = SearchParams(
        keywords="african art",
        location="Nigeria",
        date_start=1800,
        date_end=1950,
        era="ad",
        has_image=True,
        page=1,
        limit=20,
        sort_by="date",
        sort_order="asc",
    )

    print("=== Universal Museum Search ===")
    print(f"Search terms: {search_params.keywords}")
    print(f"Location: {search_params.location}")
    print(
        f"Date range: {search_params.date_start}-{search_params.date_end} {search_params.era}"
    )
    print(f"Images only: {search_params.has_image}")
    print()

    # Get URLs only (fast, no network requests)
    print("=== Generated Search URLs ===")
    urls = searcher.get_all_urls(search_params)
    for museum, url in urls.items():
        print(f"\n{museum.replace('_', ' ').title()}:")
        print(f"  {url}")

    print("\n" + "=" * 80)

    # Execute actual searches (makes network requests)
    print("\n=== Search Results ===")
    results = searcher.search_all(search_params)

    for museum_key, result in results.items():
        print(f"\n{museum_key.replace('_', ' ').title()}:")
        print(f"  Status: {'✓' if result.get('success') else '✗'}")
        print(f"  URL: {result.get('url', 'N/A')}")

        if "count" in result:
            print(f"  Results: {result['count']} objects found")
        if "error" in result:
            print(f"  Error: {result['error']}")
        if "note" in result:
            print(f"  Note: {result['note']}")


if __name__ == "__main__":
    main()
