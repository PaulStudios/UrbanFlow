import logging
from typing import List, Dict

from rtree import index

from server.utils.priority.basic import uuid_to_hash_int

logger = logging.getLogger(__name__)


class SpatialIndex:
    def __init__(self):
        self.idx = index.Index()
        self.signal_dict = {}

    def insert(self, signal_id, lat, lon):
        self.idx.insert(signal_id, (lat, lon, lat, lon))
        self.signal_dict[signal_id] = (lat, lon)

    def query(self, bbox):
        return list(self.idx.intersection(bbox))


# Global spatial index for signals
signal_index = SpatialIndex()


def initialize_spatial_index(signals: List[Dict[str, str]]):
    logger.info("Initializing spatial index for signals")
    for signal in signals:
        signal_id = uuid_to_hash_int(signal['signal_id'])
        lat, lon = float(signal['latitude']), float(signal['longitude'])
        signal_index.insert(signal_id, lat, lon)
    logger.info(f"Spatial index initialized with {len(signals)} signals")


def verify_spatial_index():
    logger.info("Verifying spatial index")
    for signal_id, (lat, lon) in signal_index.signal_dict.items():
        bbox = (lat, lon, lat, lon)
        result = signal_index.query(bbox)
        if signal_id not in result:
            logger.error(f"Signal {signal_id} not found in its own bounding box!")
    logger.info("Spatial index verification complete")
