# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ppmat.utils import logger


def get_antoine_coef(name, temperature):
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={name.lower()}&Mask=4"
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        resp = requests.get(url, stream=True, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Failed to fetch Antoine data for {name}")
            return None
            
        soup = BeautifulSoup(resp.content, "html.parser")
        table = soup.find("table", attrs={"aria-label": "Antoine Equation Parameters"})
        
        if table is None:
            logger.warning(f"No Antoine data found for {name}")
            return None
            
        rows = table.find_all("tr", class_="exp")
        temperatures, as_list, bs_list, cs_list = [], [], [], []
        
        for row in rows:
            cols = row.find_all("td")
            as_list.append(float(cols[1].text))
            bs_list.append(float(cols[2].text))
            cs_list.append(float(cols[3].text))
            
            temp_range = cols[0].text.replace(" ", "").split("-")
            temperatures.append([float(temp_range[0]), float(temp_range[1])])
        
        index = None
        for i, interval in enumerate(temperatures):
            if interval[0] <= temperature <= interval[1]:
                index = i
                break
        
        if index is None:
            raise ValueError(
                f"Temperature {temperature:.2f} K is outside the valid range for {name}"
            )
        
        return [as_list[index], bs_list[index], cs_list[index]]
        
    except Exception as e:
        logger.warning(f"Error fetching Antoine coefficients for {name}: {e}")
        return None
