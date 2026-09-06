import logging
from typing import Any, Dict, List

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataAnalyzer:
    def __init__(self):
        self._cache = {}
        self._cache_df_fingerprint = None

    def clear_cache(self):
        self._cache = {}
        self._cache_df_fingerprint = None

    def _ensure_cache_df(self, df: pd.DataFrame) -> None:
        fingerprint = (id(df), tuple(df.shape), tuple((str(c) for c in df.columns)), tuple((str(t) for t in df.dtypes.tolist())))
        if self._cache_df_fingerprint is None or self._cache_df_fingerprint != fingerprint:
            self._cache = {}
            self._cache_df_fingerprint = fingerprint

    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        self._ensure_cache_df(df)
        cache_key = ("data_quality",)
        if cache_key in self._cache:
            logger.info("使用缓存的数据质量分析结果")
            return self._cache[cache_key]
        logger.info("分析数据质量...")
        results = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": {},
            "duplicates": 0,
            "numeric_columns": [],
            "categorical_columns": [],
        }
        missing_counts = df.isnull().sum()
        missing_ratios = (missing_counts / len(df) * 100).round(2)
        results["missing_values"] = {col: {"count": int(missing_counts[col]), "ratio": float(missing_ratios[col])} for col in df.columns if missing_counts[col] > 0}
        results["duplicates"] = int(df.duplicated().sum())
        results["numeric_columns"] = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        results["categorical_columns"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._cache[cache_key] = results
        logger.info(f"数据质量分析完成: {results['shape'][0]} 行, {results['shape'][1]} 列")
        return results

    def analyze_distributions(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        self._ensure_cache_df(df)
        logger.info("分析数据分布...")
        if columns is None:
            columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cache_key = ("distributions", tuple(columns))
        if cache_key in self._cache:
            logger.info("使用缓存的数据分布分析结果")
            return self._cache[cache_key]
        results = {}
        for col in columns:
            if col not in df.columns:
                continue
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            stats_dict = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75)),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
            }
            if len(col_data) >= 3 and len(col_data) <= 5000:
                try:
                    _, p_value = stats.shapiro(col_data.sample(min(5000, len(col_data)), random_state=42))
                    stats_dict["normality_test"] = {"test": "Shapiro-Wilk", "p_value": float(p_value), "is_normal": p_value > 0.05}
                except Exception as e:
                    logger.warning(f"正态性检验失败 {col}: {e}")
            Q1 = stats_dict["q25"]
            Q3 = stats_dict["q75"]
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            stats_dict["outliers"] = {
                "count": int(len(outliers)),
                "ratio": float(len(outliers) / len(col_data) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
            results[col] = stats_dict
        self._cache[cache_key] = results
        logger.info(f"分布分析完成: {len(results)} 个特征")
        return results

    def analyze_correlations(self, df: pd.DataFrame, method: str = "pearson", threshold: float = 0.7) -> Dict[str, Any]:
        self._ensure_cache_df(df)
        cache_key = ("correlations", str(method), float(threshold))
        if cache_key in self._cache:
            logger.info("使用缓存的相关性分析结果")
            return self._cache[cache_key]
        logger.info(f"分析特征相关性 (方法: {method})...")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        if numeric_df.shape[1] < 2:
            logger.warning("数值列少于2个，无法进行相关性分析")
            return {}
        corr_matrix = numeric_df.corr(method=method)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({"feature1": corr_matrix.columns[i], "feature2": corr_matrix.columns[j], "correlation": float(corr_value)})
        results = {"correlation_matrix": corr_matrix.to_dict(), "high_correlation_pairs": high_corr_pairs, "method": method, "threshold": threshold}
        self._cache[cache_key] = results
        logger.info(f"相关性分析完成: 发现 {len(high_corr_pairs)} 对高相关特征")
        return results

    def recommend_preprocessing_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("推荐预处理策略...")
        quality = self.analyze_data_quality(df)
        distributions = self.analyze_distributions(df)
        recommendations = {
            "missing_value_strategy": {},
            "outlier_strategy": {},
            "transformation_strategy": {},
            "scaling_strategy": None,
            "reasoning": [],
        }
        for col, info in quality["missing_values"].items():
            if info["ratio"] > 50:
                recommendations["missing_value_strategy"][col] = "drop_column"
                recommendations["reasoning"].append(f"{col}: 缺失值比例 {info['ratio']:.1f}% > 50%，建议删除该列")
            elif info["ratio"] > 5:
                recommendations["missing_value_strategy"][col] = "impute_median"
                recommendations["reasoning"].append(f"{col}: 缺失值比例 {info['ratio']:.1f}%，建议使用中位数填充")
            else:
                recommendations["missing_value_strategy"][col] = "impute_mean"
                recommendations["reasoning"].append(f"{col}: 缺失值比例 {info['ratio']:.1f}%，建议使用均值填充")
        for col, dist in distributions.items():
            if "outliers" in dist and dist["outliers"]["ratio"] > 5:
                recommendations["outlier_strategy"][col] = "clip"
                recommendations["reasoning"].append(f"{col}: 异常值比例 {dist['outliers']['ratio']:.1f}%，建议使用 IQR 方法裁剪")
        for col, dist in distributions.items():
            if abs(dist["skewness"]) > 1:
                if dist["min"] > 0:
                    recommendations["transformation_strategy"][col] = "log_transform"
                    recommendations["reasoning"].append(f"{col}: 偏度 {dist['skewness']:.2f}，建议进行对数转换")
                else:
                    recommendations["transformation_strategy"][col] = "box_cox"
                    recommendations["reasoning"].append(f"{col}: 偏度 {dist['skewness']:.2f}，建议进行 Box-Cox 转换")
        ranges = []
        for col, dist in distributions.items():
            ranges.append(dist["max"] - dist["min"])
        if len(ranges) > 0:
            max_range = max(ranges)
            min_range = min(ranges)
            if max_range / (min_range + 1e-09) > 100:
                recommendations["scaling_strategy"] = "standard_scaler"
                recommendations["reasoning"].append(f"特征范围差异大 (最大/最小 = {max_range / min_range:.1f})，建议使用 StandardScaler")
            else:
                recommendations["scaling_strategy"] = "minmax_scaler"
                recommendations["reasoning"].append("特征范围相对一致，建议使用 MinMaxScaler")
        logger.info(f"预处理策略推荐完成: {len(recommendations['reasoning'])} 条建议")
        return recommendations

    def generate_data_report(self, df: pd.DataFrame, *, include_correlations: bool = True) -> str:
        logger.info("生成数据分析报告...")
        quality = self.analyze_data_quality(df)
        distributions = self.analyze_distributions(df)
        correlations: Dict[str, Any] = {}
        if include_correlations:
            correlations = self.analyze_correlations(df)
        recommendations = self.recommend_preprocessing_strategy(df)
        report = []
        report.append("=" * 80)
        report.append("数据分析报告")
        report.append("=" * 80)
        report.append("")
        report.append("## 1. 数据概览")
        report.append(f"- 数据形状: {quality['shape'][0]} 行 × {quality['shape'][1]} 列")
        report.append(f"- 数值列: {len(quality['numeric_columns'])} 个")
        report.append(f"- 分类列: {len(quality['categorical_columns'])} 个")
        report.append(f"- 重复行: {quality['duplicates']} 行")
        report.append("")
        report.append("## 2. 数据质量")
        if quality["missing_values"]:
            report.append("### 缺失值:")
            for col, info in quality["missing_values"].items():
                report.append(f"  - {col}: {info['count']} ({info['ratio']:.1f}%)")
        else:
            report.append("- 无缺失值")
        report.append("")
        report.append("## 3. 分布特征")
        for i, (col, dist) in enumerate(list(distributions.items())):
            report.append(f"### {col}:")
            report.append(f"  - 均值: {dist['mean']:.4f}, 中位数: {dist['median']:.4f}")
            report.append(f"  - 标准差: {dist['std']:.4f}")
            report.append(f"  - 范围: [{dist['min']:.4f}, {dist['max']:.4f}]")
            report.append(f"  - 偏度: {dist['skewness']:.2f}, 峰度: {dist['kurtosis']:.2f}")
            if "outliers" in dist:
                report.append(f"  - 异常值: {dist['outliers']['count']} ({dist['outliers']['ratio']:.1f}%)")
            report.append("")
        report.append("## 4. 高相关性特征对")
        if not include_correlations:
            report.append("- 本阶段未执行相关性分析（将在后续特征分析阶段进行）")
        else:
            if correlations.get("high_correlation_pairs"):
                for pair in correlations["high_correlation_pairs"]:
                    report.append(f"  - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
            else:
                report.append("- 无高相关性特征对")
        report.append("")
        report.append("## 5. 预处理建议")
        for reason in recommendations["reasoning"]:
            report.append(f"  - {reason}")
        report.append("")
        report.append("=" * 80)
        report_text = "\n".join(report)
        logger.info("数据分析报告生成完成")
        return report_text

