"""MDF data service for efficient signal access"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("asammdf.plugins.aiaspro.services")


class SignalData:
    """Container for signal data and metadata"""
    
    def __init__(self, name: str, timestamps=None, samples=None, 
                 unit: str = "", metadata: Dict = None):
        """Initialize signal data
        
        Args:
            name: Signal name
            timestamps: Time array (optional)
            samples: Sample values (optional)
            unit: Signal unit
            metadata: Additional metadata
        """
        self.name = name
        self.timestamps = timestamps
        self.samples = samples
        self.unit = unit
        self.metadata = metadata or {}
        
        # Computed properties (lazy evaluation)
        self._statistics = None
        self._sample_rate = None
        self._duration = None
    
    @property
    def statistics(self) -> Dict[str, float]:
        """Get signal statistics (lazy computed)"""
        if self._statistics is None and self.samples is not None:
            try:
                import numpy as np
                self._statistics = {
                    "min": float(np.min(self.samples)),
                    "max": float(np.max(self.samples)),
                    "mean": float(np.mean(self.samples)),
                    "std": float(np.std(self.samples)),
                    "count": len(self.samples)
                }
            except Exception as e:
                logger.warning(f"Error computing statistics for {self.name}: {e}")
                self._statistics = {}
        
        return self._statistics or {}
    
    @property
    def sample_rate(self) -> float:
        """Get average sample rate"""
        if self._sample_rate is None and self.timestamps is not None:
            try:
                if len(self.timestamps) > 1:
                    duration = self.timestamps[-1] - self.timestamps[0]
                    self._sample_rate = len(self.timestamps) / duration if duration > 0 else 0
                else:
                    self._sample_rate = 0
            except Exception:
                self._sample_rate = 0
        
        return self._sample_rate or 0
    
    @property
    def duration(self) -> float:
        """Get signal duration in seconds"""
        if self._duration is None and self.timestamps is not None:
            try:
                if len(self.timestamps) > 1:
                    self._duration = self.timestamps[-1] - self.timestamps[0]
                else:
                    self._duration = 0
            except Exception:
                self._duration = 0
        
        return self._duration or 0


class MDFDataService:
    """Service for accessing MDF data efficiently
    
    This service provides high-level access to MDF data with caching
    and intelligent signal management.
    """
    
    def __init__(self, mdf):
        """Initialize MDF data service
        
        Args:
            mdf: MDF object from asammdf
        """
        self.mdf = mdf
        self._signal_cache = {}
        self._metadata_cache = {}
        self._categories_cache = None
        
        logger.info("MDF Data Service initialized")
    
    def get_channels(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of channels, optionally filtered
        
        Args:
            pattern: Optional pattern to filter channel names
            
        Returns:
            List of channel names
        """
        try:
            if hasattr(self.mdf, 'channels_db'):
                channels = list(self.mdf.channels_db)
            else:
                logger.warning("MDF object has no channels_db attribute")
                return []
            
            if pattern:
                pattern_lower = pattern.lower()
                channels = [ch for ch in channels if pattern_lower in ch.lower()]
            
            return channels
            
        except Exception as e:
            logger.error(f"Error getting channels: {e}")
            return []
    
    def load_signal(self, channel_name: str, use_cache: bool = True) -> Optional[SignalData]:
        """Load a single signal with caching
        
        Args:
            channel_name: Name of the channel to load
            use_cache: Whether to use cached data
            
        Returns:
            SignalData object or None if error
        """
        # Check cache first
        if use_cache and channel_name in self._signal_cache:
            return self._signal_cache[channel_name]
        
        try:
            # Get signal from MDF
            signal = self.mdf.get(channel_name)
            
            # Create SignalData object
            signal_data = SignalData(
                name=channel_name,
                timestamps=signal.timestamps,
                samples=signal.samples,
                unit=signal.unit or "",
                metadata=self._extract_metadata(signal)
            )
            
            # Cache the result
            if use_cache:
                self._signal_cache[channel_name] = signal_data
            
            logger.debug(f"Loaded signal: {channel_name} ({len(signal.samples)} samples)")
            return signal_data
            
        except Exception as e:
            logger.error(f"Error loading signal '{channel_name}': {e}")
            return None
    
    def load_signals(self, patterns: List[str], use_cache: bool = True) -> Dict[str, SignalData]:
        """Load multiple signals by patterns
        
        Args:
            patterns: List of patterns to match channel names
            use_cache: Whether to use cached data
            
        Returns:
            Dict mapping channel names to SignalData objects
        """
        signals = {}
        
        for pattern in patterns:
            matching_channels = self.get_channels(pattern)
            
            for channel in matching_channels:
                signal_data = self.load_signal(channel, use_cache)
                if signal_data:
                    signals[channel] = signal_data
        
        logger.info(f"Loaded {len(signals)} signals for patterns: {patterns}")
        return signals
    
    def get_channel_categories(self) -> Dict[str, List[str]]:
        """Categorize channels by naming patterns and metadata
        
        Returns:
            Dict mapping category names to channel lists
        """
        if self._categories_cache is not None:
            return self._categories_cache
        
        channels = self.get_channels()
        categories = defaultdict(list)
        
        # Automotive-specific categorization
        automotive_categories = {
            "Engine": ["engine", "rpm", "throttle", "torque", "power", "ignition"],
            "Transmission": ["gear", "transmission", "clutch", "shift"],
            "Braking": ["brake", "abs", "pressure", "pedal"],
            "Vehicle_Dynamics": ["speed", "accel", "velocity", "wheel", "steering"],
            "Fuel": ["fuel", "consumption", "injection", "pump"],
            "Temperature": ["temp", "cool", "thermal", "heat"],
            "Electrical": ["volt", "amp", "battery", "current", "charge"],
            "Exhaust": ["exhaust", "emission", "lambda", "oxygen"],
            "CAN_Bus": ["can", "lin", "bus", "network"],
            "Sensors": ["sensor", "position", "angle", "level"]
        }
        
        # Categorize by automotive terms
        for channel in channels:
            channel_lower = channel.lower()
            categorized = False
            
            for category, terms in automotive_categories.items():
                if any(term in channel_lower for term in terms):
                    categories[category].append(channel)
                    categorized = True
                    break
            
            # Categorize by prefix (e.g., "ECM.EngineSpeed" -> "ECM")
            if not categorized:
                if '.' in channel:
                    prefix = channel.split('.')[0]
                    categories[f"ECU_{prefix}"].append(channel)
                    categorized = True
            
            # Uncategorized signals
            if not categorized:
                categories["Other"].append(channel)
        
        # Remove empty categories and sort
        self._categories_cache = {
            category: sorted(channels) 
            for category, channels in categories.items() 
            if channels
        }
        
        logger.info(f"Categorized {len(channels)} channels into {len(self._categories_cache)} categories")
        return self._categories_cache
    
    def search_channels(self, query: str, fuzzy: bool = True) -> List[str]:
        """Intelligent channel search with fuzzy matching
        
        Args:
            query: Search query
            fuzzy: Whether to use fuzzy matching
            
        Returns:
            List of matching channel names
        """
        channels = self.get_channels()
        query_lower = query.lower()
        
        if not fuzzy:
            return [ch for ch in channels if query_lower in ch.lower()]
        
        # Fuzzy search with scoring
        matches = []
        
        for channel in channels:
            channel_lower = channel.lower()
            score = 0
            
            # Exact substring match (highest score)
            if query_lower in channel_lower:
                score += 100
            
            # Word boundary matches
            for word in query_lower.split():
                if word in channel_lower:
                    score += 50
            
            # Character sequence matches
            query_chars = list(query_lower)
            channel_chars = list(channel_lower)
            char_matches = sum(1 for c in query_chars if c in channel_chars)
            score += char_matches
            
            if score > 0:
                matches.append((channel, score))
        
        # Sort by score and return channel names
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches]
    
    def get_signal_summary(self, channel_name: str) -> Dict[str, Any]:
        """Get summary information for a signal
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Summary information dict
        """
        signal_data = self.load_signal(channel_name)
        
        if not signal_data:
            return {"error": f"Could not load signal '{channel_name}'"}
        
        summary = {
            "name": signal_data.name,
            "unit": signal_data.unit,
            "sample_count": len(signal_data.samples) if signal_data.samples is not None else 0,
            "duration": signal_data.duration,
            "sample_rate": signal_data.sample_rate,
            "statistics": signal_data.statistics,
            "metadata": signal_data.metadata
        }
        
        return summary
    
    def find_correlated_signals(self, reference_signal: str, 
                               threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find signals correlated with a reference signal
        
        Args:
            reference_signal: Name of the reference signal
            threshold: Correlation threshold (0-1)
            
        Returns:
            List of (channel_name, correlation) tuples
        """
        ref_data = self.load_signal(reference_signal)
        if not ref_data or ref_data.samples is None:
            return []
        
        try:
            import numpy as np
            correlations = []
            
            # Get a sample of other channels for correlation analysis
            channels = self.get_channels()
            sample_channels = channels[:50]  # Limit for performance
            
            for channel in sample_channels:
                if channel == reference_signal:
                    continue
                
                signal_data = self.load_signal(channel)
                if not signal_data or signal_data.samples is None:
                    continue
                
                try:
                    # Align signals if needed and compute correlation
                    ref_samples, signal_samples = self._align_signals(
                        ref_data.samples, signal_data.samples
                    )
                    
                    correlation = np.corrcoef(ref_samples, signal_samples)[0, 1]
                    
                    if abs(correlation) >= threshold:
                        correlations.append((channel, float(correlation)))
                        
                except Exception as e:
                    logger.debug(f"Error computing correlation for {channel}: {e}")
                    continue
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            return correlations
            
        except Exception as e:
            logger.error(f"Error finding correlated signals: {e}")
            return []
    
    def _extract_metadata(self, signal) -> Dict[str, Any]:
        """Extract metadata from a signal object
        
        Args:
            signal: Signal object from asammdf
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        try:
            # Extract common metadata fields
            if hasattr(signal, 'comment') and signal.comment:
                metadata['comment'] = signal.comment
            
            if hasattr(signal, 'unit') and signal.unit:
                metadata['unit'] = signal.unit
            
            if hasattr(signal, 'source') and signal.source:
                metadata['source'] = str(signal.source)
            
            # Add more metadata extraction as needed
            
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _align_signals(self, signal1, signal2):
        """Align two signals for comparison
        
        Args:
            signal1: First signal array
            signal2: Second signal array
            
        Returns:
            Tuple of aligned signal arrays
        """
        try:
            import numpy as np
            
            # Simple alignment - truncate to shorter length
            min_length = min(len(signal1), len(signal2))
            return signal1[:min_length], signal2[:min_length]
            
        except Exception:
            return signal1, signal2
    
    def clear_cache(self):
        """Clear all cached data"""
        self._signal_cache.clear()
        self._metadata_cache.clear()
        self._categories_cache = None
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache usage
        
        Returns:
            Cache information dict
        """
        return {
            "cached_signals": len(self._signal_cache),
            "cached_metadata": len(self._metadata_cache),
            "categories_cached": self._categories_cache is not None
        }