# data_context_quality.py

import json
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI

KNOWLEDGE_BASE: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    # 电力领域 (Electricity)
    "electricity": {
        "Demand": {
            "Load": ["load"],
            "Weather": ["temperature", "humidity", "wind_speed", "solar_irradiance"],
            "Calendar": ["day_of_week", "hour_of_day", "month", "holiday", "is_weekend"]
        },
        "Supply": {
            "Total_Generation": ["generation_total"],
            "Generation_Mix": ["generation_solar", "generation_wind", "generation_hydro", "generation_gas",
                               "generation_coal", "generation_nuclear"],
            "Self_Supply": ["self_supplied_unit", "local power generation"]
        },
        "Grid": {
            "System_State": ["grid_frequency", "positive_backup", "negative_backup"],
            "Cross_Region_Flow": ["foreign_electricity_trading"]
        },
        "Market": {
            "Price": ["price"]
        }
    },
    # 金融领域 (Finance)
    "finance": {
        "Market_Price": {
            "Price_Action": ["open_price", "close_price", "high_price", "low_price"],
            "Technical_Indicators": ["moving_average_50d", "moving_average_200d", "rsi_14d", "macd"]
        },
        "Market_Activity": {
            "Volume_Turnover": ["volume", "turnover_rate"],
            "Volatility": ["volatility_index_vix"]
        },
        "Macroeconomics": {
            "Economic_Indicators": ["gdp_growth_rate", "unemployment_rate"],
            "Monetary_Policy": ["interest_rate", "inflation_rate_cpi"]
        },
        "Market_Sentiment": {
            "Broad_Market": ["market_index_close"],
            "News": ["news_sentiment_score"]
        }
    },
    # 交通领域 (Traffic)
    "traffic": {
        "Flow_Metrics": {
            "Volume": ["vehicle_count"],
            "Speed": ["average_speed"],
            "Congestion": ["road_occupancy", "travel_time", "traffic_congestion_index"]
        },
        "Influencing_Factors": {
            "Calendar": ["day_of_week", "is_peak_hour", "is_holiday"],
            "Environment": ["weather_condition", "temperature", "precipitation"],
            "Events": ["accident_count", "is_special_event", "roadwork_active"]
        }
    }
}


def score_calculate_domain_diversity(list_of_dfs: List[pd.DataFrame], api_key: str) -> float:
    if not list_of_dfs:
        raise ValueError("Input list of DataFrames cannot be empty.")

    main_df = list_of_dfs[0]
    present_columns = main_df.columns.tolist()

    # 1. 使用LLM识别当前数据集的领域
    available_domains = list(KNOWLEDGE_BASE.keys())
    try:
        identified_domain = get_domain_from_llm(present_columns, available_domains, api_key)
    except Exception:
        print("Diversity Score: Could not determine domain. Returning a score of 0.")
        return 0.0

    domain_kb = KNOWLEDGE_BASE.get(identified_domain)
    if not domain_kb:
        print(f"Diversity Score: No knowledge base found for domain '{identified_domain}'.")
        return 100.0

    # 2. 从知识库中提取所有的二级维度作为评估基准
    all_sub_facets = []
    for primary_facet, sub_facets_dict in domain_kb.items():
        all_sub_facets.extend(sub_facets_dict.keys())

    if not all_sub_facets:
        return 100.0  # 如果知识库中没有定义二级维度，则认为多样性是满足的

    total_sub_facets_count = len(all_sub_facets)

    # 3. 遍历每个二级维度，使用LLM检查其是否被当前数据集的特征所覆盖
    covered_sub_facets_count = 0
    for primary_facet, sub_facets_dict in domain_kb.items():
        for sub_facet_name, canonical_features in sub_facets_dict.items():
            try:
                # 调用LLM进行智能匹配，判断该维度下的标准特征是否存在
                matches = match_features_with_llm(present_columns, canonical_features, api_key)
                if matches > 0:
                    # 只要此维度的标准特征列表中，有至少一个特征被匹配到，就认为该维度被覆盖
                    covered_sub_facets_count += 1
            except Exception as e:
                print(
                    f"Diversity Score: An error occurred during feature matching for sub-facet '{sub_facet_name}': {e}")
                continue

    # 4. 计算最终的维度覆盖广度得分
    score = (covered_sub_facets_count / total_sub_facets_count) * 100
    return score


def score_calculate_domain_completeness(list_of_dfs: List[pd.DataFrame], api_key: str) -> float:
    if not list_of_dfs:
        raise ValueError("Input list of DataFrames cannot be empty.")

    main_df = list_of_dfs[0]
    present_columns = main_df.columns.tolist()

    available_domains = list(KNOWLEDGE_BASE.keys())
    try:
        identified_domain = get_domain_from_llm(present_columns, available_domains, api_key)
    except Exception:
        print("Completeness Score: Could not determine domain. Returning a score of 0.")
        return 0.0

    domain_kb = KNOWLEDGE_BASE.get(identified_domain)
    if not domain_kb:
        print(f"Completeness Score: No knowledge base found for domain '{identified_domain}'.")
        return 100.0

    required_features = []
    for primary_facet, sub_facets_dict in domain_kb.items():
        for sub_facet_name, canonical_features in sub_facets_dict.items():
            required_features.extend(canonical_features)

    required_features = sorted(list(set(required_features)))

    if not required_features:
        print(f"No features found in the knowledge base for domain '{identified_domain}'.")
        return 100.0

    N_kb = len(required_features)

    # 3. 智能特征匹配 (无变化)
    try:
        N_s_prime = match_features_with_llm(present_columns, required_features, api_key)
    except Exception:
        print("Completeness Score: Could not perform feature matching. Returning a score of 0.")
        return 0.0

    # 4. 计算最终得分 (无变化)
    if N_kb == 0:
        score = 100.0
    else:
        score = (N_s_prime / N_kb) * 100

    return score


def get_domain_from_llm(columns: List[str], available_domains: List[str], api_key: str) -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt_content = f"""
    Analyze the following list of column names from a time series dataset and identify which domain it belongs to.

    Column names: {columns}

    Please choose the single best domain from this list: {available_domains}.

    Your response MUST be only one word from the provided list and nothing else.
    """

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant specializing in time series analysis. Your task is to identify the domain of a dataset based on its column names."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0
        )
        identified_domain = completion.choices[0].message.content.strip().lower()

        if identified_domain not in available_domains:
            # 如果LLM未能返回有效领域，增加一个简单的后备匹配逻辑
            for domain in available_domains:
                if domain in identified_domain:
                    return domain
            raise ValueError(
                f"LLM returned an invalid domain '{identified_domain}', which is not in the knowledge base.")

        return identified_domain

    except Exception as e:
        print(f"An error occurred while communicating with LLM: {e}")
        raise


def match_features_with_llm(present_columns: List[str], required_features: List[str], api_key: str) -> int:
    if not required_features:
        return 0
    if not present_columns:
        return 0

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt_content = f"""
    You are an expert in data science. Your task is to determine which of the required canonical features are present in a given list of dataset columns. The column names may be synonyms or variations.

    Here are the columns from the user's dataset:
    {present_columns}

    Here is the list of canonical features that are considered essential:
    {required_features}

    Your response must be a JSON object. The keys of the JSON should be the canonical feature names from the essential list. The value for each key should be the matching column name from the user's dataset, or null if no appropriate column is found.

    Example Response Format:
    {{
      "load": "power_consumption_kw",
      "price": "grid_price",
      "temperature": "ambient_temp_c",
      "holiday": null
    }}
    """
    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides responses in JSON format."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0
        )
        response_text = completion.choices[0].message.content
        matching_map = json.loads(response_text)
        # 统计 JSON 中值不为 null 的项的数量，即为成功匹配的数量
        successful_matches = sum(1 for value in matching_map.values() if value is not None)
        return successful_matches

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from LLM response.")
        return 0
    except Exception as e:
        print(f"An error occurred during feature matching: {e}")
        raise