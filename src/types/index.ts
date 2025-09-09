export interface ForecastData {
  product?: string;
  productGroup?: string;
  productHierarchy?: string;
  location?: string;
  locationRegion?: string;
  customer?: string;
  customerGroup?: string;
  customerRegion?: string;
  shipToParty?: string;
  soldToParty?: string;
  uom?: string;
  date: string;
  unitPrice?: number;
  quantity: number;
}

export interface AggregatedData {
  date: string;
  quantity: number;
  period: string;
}

export interface ForecastConfig {
  forecastBy: 'product' | 'customer' | 'location';
  selectedItem: string;
  selectedProduct?: string;
  selectedCustomer?: string;
  selectedLocation?: string;
  interval: 'week' | 'month' | 'year';
  historicPeriod: number;
  forecastPeriod: number;
}
