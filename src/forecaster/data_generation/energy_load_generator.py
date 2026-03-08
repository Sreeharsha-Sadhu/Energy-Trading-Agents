import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import (
    BASE_CONSUMPTION,
    DEFAULT_OUTPUT_CSV,
    LOAD_PROFILES,
    LOSS_FACTORS,
    SOLAR_CAPACITY,
    TRACKING_FILE,
    TZ,
)
from .load_factors import (
    generate_meter_count,
    get_hourly_load_factor,
    get_seasonal_factor,
    get_weekend_factor,
)
from .solar_model import get_solar_irradiance
from .tracking import load_tracking_data, save_tracking_data
from .writer import write_data_chunk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EnergyLoadDataGenerator:
    """Orchestrates generation of hourly load records with solar logic."""

    def __init__(
        self, tracking_file: str = TRACKING_FILE, output_csv: str = DEFAULT_OUTPUT_CSV
    ):
        """Init."""
        self.tz = TZ
        self.tracking_file = tracking_file
        self.output_csv = output_csv
        self.load_profiles = LOAD_PROFILES
        self.base_consumption = BASE_CONSUMPTION
        self.solar_capacity = SOLAR_CAPACITY
        self.loss_factors = LOSS_FACTORS

    def get_business_days_ago(
        self, current_date: datetime.date, business_days: int
    ) -> datetime.date:
        """Get Business Days Ago."""
        check_date = current_date
        while business_days > 0:
            check_date -= timedelta(days=1)
            if check_date.weekday() < 5:
                business_days -= 1
        return check_date

    def calculate_data_availability_dates(self, current_date: datetime.date):
        """Calculate Data Availability Dates."""
        initial_cutoff = self.get_business_days_ago(current_date, 7)
        final_cutoff = self.get_business_days_ago(current_date, 48)
        return initial_cutoff, final_cutoff

    def generate_hourly_data(
        self, dt, load_profile: str, rate_group: str, is_solar: bool, year_index: int
    ) -> Dict[str, Any]:
        """Generate Hourly Data."""
        base_params = self.base_consumption[load_profile]
        base_meters = base_params["meters_per_group"]
        growth_factor = (1.02) ** year_index
        current_meters = int(base_meters * growth_factor)

        load_meter_count = generate_meter_count(current_meters)
        gen_meter_count = int(load_meter_count * 0.4) if is_solar else 0
        total_meter_count = load_meter_count + gen_meter_count

        hour_factor = get_hourly_load_factor(dt.hour, load_profile)
        seasonal_factor = get_seasonal_factor(dt.month)
        weekend_factor = get_weekend_factor(dt.weekday())

        base_load_per_meter = np.random.normal(base_params["avg"], base_params["std"])
        total_base_load = (
            base_load_per_meter
            * load_meter_count
            * hour_factor
            * seasonal_factor
            * weekend_factor
        )
        total_base_load *= np.random.normal(1.0, 0.1)

        gen_base_load = 0.0
        if is_solar and gen_meter_count > 0:
            solar_irradiance = get_solar_irradiance(dt)
            solar_params = self.solar_capacity[load_profile]
            avg_solar_per_meter = np.random.normal(
                solar_params["avg"], solar_params["std"]
            )
            gen_base_load = -(
                avg_solar_per_meter
                * gen_meter_count
                * solar_irradiance
                * np.random.normal(1.0, 0.1)
            )

        loss_factor = self.loss_factors[load_profile]
        load_lal = total_base_load * (1 + loss_factor)
        gen_lal = gen_base_load * (1 + loss_factor) if gen_base_load != 0 else 0.0

        base_load = total_base_load + gen_base_load
        loss_adjusted_load = load_lal + gen_lal

        return {
            "BaseLoad": round(base_load, 2),
            "LossAdjustedLoad": round(loss_adjusted_load, 2),
            "MeterCount": total_meter_count,
            "LoadBL": round(total_base_load, 2),
            "LoadLAL": round(load_lal, 2),
            "LoadMeterCount": load_meter_count,
            "GenBL": round(gen_base_load, 2),
            "GenLAL": round(gen_lal, 2),
            "GenMeterCount": gen_meter_count,
        }

    def load_tracking(self):
        """Load Tracking."""
        return load_tracking_data(self.tracking_file)

    def save_tracking(self, tracking_data: Dict[str, Any]):
        """Save Tracking."""
        save_tracking_data(self.tracking_file, tracking_data)

    def generate_incremental_dataset(
        self,
        current_date: Optional[datetime] = None,
        force_full_regeneration: bool = False,
        chunk_size: int = 50000,
    ):
        """Generate Incremental Dataset."""
        if current_date is None:
            current_date = datetime.now().date()
        elif isinstance(current_date, datetime):
            current_date = current_date.date()

        tracking_data = self.load_tracking()
        initial_cutoff, final_cutoff = self.calculate_data_availability_dates(
            current_date
        )

        logger.info("Current date: %s", current_date)
        logger.info(
            "Initial cutoff: %s | Final cutoff: %s", initial_cutoff, final_cutoff
        )

        data_chunk = []
        record_id = 1
        total_records = 0
        output_filename = self.output_csv
        is_first_chunk = True

        if force_full_regeneration or tracking_data.get("first_run", True):
            start_date_initial = datetime(2020, 1, 1).date()
            start_date_final = datetime(2020, 1, 1).date()
            logger.info("First run - full regeneration.")
            if os.path.exists(output_filename):
                os.remove(output_filename)
                logger.info("Removed existing file: %s", output_filename)
        else:
            if os.path.exists(output_filename):
                try:
                    existing_df = pd.read_csv(output_filename, usecols=["Id"])
                    if len(existing_df) > 0:
                        record_id = int(existing_df["Id"].max()) + 1
                        total_records = len(existing_df)
                        is_first_chunk = False
                        logger.info(
                            "Loaded existing data info: %d records, starting from ID %d",
                            total_records,
                            record_id,
                        )
                except Exception as e:
                    logger.warning("Could not read existing CSV for ID: %s", e)
                    record_id = 1

            last_initial = (
                pd.to_datetime(tracking_data.get("last_initial_date"))
                if tracking_data.get("last_initial_date")
                else pd.to_datetime("2020-01-01")
            ).date()
            last_final = (
                pd.to_datetime(tracking_data.get("last_final_date"))
                if tracking_data.get("last_final_date")
                else pd.to_datetime("2020-01-01")
            ).date()

            start_date_initial = last_initial + timedelta(days=1)
            start_date_final = last_final + timedelta(days=1)

            logger.info(
                "Incremental run - Initial: %s to %s; Final: %s to %s",
                start_date_initial,
                initial_cutoff,
                start_date_final,
                final_cutoff,
            )

        def _generate_for_range(start_date, end_date, submission_type, accuracy_sd):
            """Generate For Range."""
            nonlocal data_chunk, record_id, total_records, is_first_chunk
            current_gen_date = start_date
            while current_gen_date <= end_date:
                year_index = current_gen_date.year - 2020
                for hour in range(24):
                    dt = datetime.combine(
                        current_gen_date, datetime.min.time().replace(hour=hour)
                    )
                    dt_pst = self.tz.localize(dt)
                    for load_profile, rate_groups in self.load_profiles.items():
                        for rate_group in rate_groups.get("non_solar", []):
                            hourly = self.generate_hourly_data(
                                dt_pst, load_profile, rate_group, False, year_index
                            )
                            accuracy_factor = np.random.normal(1.0, accuracy_sd)
                            record = {
                                "Id": record_id,
                                "TradeDate": current_gen_date.strftime("%Y-%m-%d"),
                                "TradeTime": f"{hour:02d}:00",
                                "LoadProfile": load_profile,
                                "RateGroup": rate_group,
                                "BaseLoad": round(
                                    hourly["BaseLoad"] * accuracy_factor, 2
                                ),
                                "LossAdjustedLoad": round(
                                    hourly["LossAdjustedLoad"] * accuracy_factor, 2
                                ),
                                "MeterCount": hourly["MeterCount"],
                                "LoadBL": round(hourly["LoadBL"] * accuracy_factor, 2),
                                "LoadLAL": round(
                                    hourly["LoadLAL"] * accuracy_factor, 2
                                ),
                                "LoadMeterCount": hourly["LoadMeterCount"],
                                "GenBL": 0.0,
                                "GenLAL": 0.0,
                                "GenMeterCount": 0,
                                "Submission": submission_type,
                                "Created": (dt_pst + timedelta(hours=1)).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                            data_chunk.append(record)
                            record_id += 1
                            total_records += 1

                        for rate_group in rate_groups.get("solar", []):
                            hourly = self.generate_hourly_data(
                                dt_pst, load_profile, rate_group, True, year_index
                            )
                            accuracy_factor = np.random.normal(1.0, accuracy_sd)
                            record = {
                                "Id": record_id,
                                "TradeDate": current_gen_date.strftime("%Y-%m-%d"),
                                "TradeTime": f"{hour:02d}:00",
                                "LoadProfile": load_profile,
                                "RateGroup": rate_group,
                                "BaseLoad": round(
                                    hourly["BaseLoad"] * accuracy_factor, 2
                                ),
                                "LossAdjustedLoad": round(
                                    hourly["LossAdjustedLoad"] * accuracy_factor, 2
                                ),
                                "MeterCount": hourly["MeterCount"],
                                "LoadBL": round(hourly["LoadBL"] * accuracy_factor, 2),
                                "LoadLAL": round(
                                    hourly["LoadLAL"] * accuracy_factor, 2
                                ),
                                "LoadMeterCount": hourly["LoadMeterCount"],
                                "GenBL": round(hourly["GenBL"] * accuracy_factor, 2),
                                "GenLAL": round(hourly["GenLAL"] * accuracy_factor, 2),
                                "GenMeterCount": hourly["GenMeterCount"],
                                "Submission": submission_type,
                                "Created": (dt_pst + timedelta(hours=1)).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                            data_chunk.append(record)
                            record_id += 1
                            total_records += 1

                    if len(data_chunk) >= chunk_size:
                        write_data_chunk(data_chunk, output_filename, is_first_chunk)
                        is_first_chunk = False
                        data_chunk = []

                current_gen_date += timedelta(days=1)

        if start_date_final <= final_cutoff:
            logger.info(
                "Generating Final submissions: %s -> %s", start_date_final, final_cutoff
            )
            _generate_for_range(start_date_final, final_cutoff, "Final", 0.02)

        if start_date_initial <= initial_cutoff:
            logger.info(
                "Generating Initial submissions: %s -> %s",
                start_date_initial,
                initial_cutoff,
            )
            _generate_for_range(start_date_initial, initial_cutoff, "Initial", 0.08)

        if data_chunk:
            write_data_chunk(data_chunk, output_filename, is_first_chunk)

        tracking_data["last_initial_date"] = initial_cutoff.strftime("%Y-%m-%d")
        tracking_data["last_final_date"] = final_cutoff.strftime("%Y-%m-%d")
        tracking_data["first_run"] = False
        tracking_data["last_run_date"] = current_date.strftime("%Y-%m-%d")
        self.save_tracking(tracking_data)

        try:
            df_summary = pd.read_csv(output_filename, nrows=5)
            logger.info(
                "Dataset generation complete — total records (approx): %d",
                total_records,
            )
            return df_summary
        except Exception as e:
            logger.warning("Could not read back summary: %s", e)
            return None
