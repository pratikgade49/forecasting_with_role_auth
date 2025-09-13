#!/usr/bin/env python3
"""
Database utility functions for optimized data retrieval
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from database import ProductCustomerLocationCombination, ForecastData
from typing import Optional, List, Dict, Any, Tuple

def get_or_create_combination(
    db: Session,
    product: Optional[str] = None,
    customer: Optional[str] = None,
    location: Optional[str] = None,
    product_group: Optional[str] = None,
    product_hierarchy: Optional[str] = None,
    location_region: Optional[str] = None,
    customer_group: Optional[str] = None,
    customer_region: Optional[str] = None,
    ship_to_party: Optional[str] = None,
    sold_to_party: Optional[str] = None
) -> int:
    """
    Get existing combination ID or create a new one
    Returns the combination_id
    """
    # Try to find existing combination
    existing = db.query(ProductCustomerLocationCombination).filter(
        and_(
            ProductCustomerLocationCombination.product == product,
            ProductCustomerLocationCombination.customer == customer,
            ProductCustomerLocationCombination.location == location
        )
    ).first()
    
    if existing:
        # Update additional fields if they're provided and different
        updated = False
        if product_group and existing.product_group != product_group:
            existing.product_group = product_group
            updated = True
        if product_hierarchy and existing.product_hierarchy != product_hierarchy:
            existing.product_hierarchy = product_hierarchy
            updated = True
        if location_region and existing.location_region != location_region:
            existing.location_region = location_region
            updated = True
        if customer_group and existing.customer_group != customer_group:
            existing.customer_group = customer_group
            updated = True
        if customer_region and existing.customer_region != customer_region:
            existing.customer_region = customer_region
            updated = True
        if ship_to_party and existing.ship_to_party != ship_to_party:
            existing.ship_to_party = ship_to_party
            updated = True
        if sold_to_party and existing.sold_to_party != sold_to_party:
            existing.sold_to_party = sold_to_party
            updated = True
            
        if updated:
            db.commit()
            
        return existing.id
    
    # Create new combination
    new_combination = ProductCustomerLocationCombination(
        product=product,
        customer=customer,
        location=location,
        product_group=product_group,
        product_hierarchy=product_hierarchy,
        location_region=location_region,
        customer_group=customer_group,
        customer_region=customer_region,
        ship_to_party=ship_to_party,
        sold_to_party=sold_to_party
    )
    
    db.add(new_combination)
    db.flush()  # Get the ID without committing the transaction
    return new_combination.id

def get_unique_values(db: Session) -> Dict[str, List[str]]:
    """
    Get unique products, customers, and locations from combinations table
    """
    # Get unique products
    products = db.query(ProductCustomerLocationCombination.product)\
        .filter(ProductCustomerLocationCombination.product.isnot(None))\
        .distinct().all()
    products = [p[0] for p in products if p[0]]
    
    # Get unique customers
    customers = db.query(ProductCustomerLocationCombination.customer)\
        .filter(ProductCustomerLocationCombination.customer.isnot(None))\
        .distinct().all()
    customers = [c[0] for c in customers if c[0]]
    
    # Get unique locations
    locations = db.query(ProductCustomerLocationCombination.location)\
        .filter(ProductCustomerLocationCombination.location.isnot(None))\
        .distinct().all()
    locations = [l[0] for l in locations if l[0]]
    
    return {
        'products': sorted(products),
        'customers': sorted(customers),
        'locations': sorted(locations)
    }

def get_filtered_combinations(
    db: Session,
    selected_products: Optional[List[str]] = None,
    selected_customers: Optional[List[str]] = None,
    selected_locations: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Get filtered unique values based on selected filters
    """
    query = db.query(ProductCustomerLocationCombination)
    
    # Apply filters
    if selected_products:
        query = query.filter(ProductCustomerLocationCombination.product.in_(selected_products))
    if selected_customers:
        query = query.filter(ProductCustomerLocationCombination.customer.in_(selected_customers))
    if selected_locations:
        query = query.filter(ProductCustomerLocationCombination.location.in_(selected_locations))
    
    combinations = query.all()
    
    # Extract unique values from filtered combinations
    products = list(set([c.product for c in combinations if c.product]))
    customers = list(set([c.customer for c in combinations if c.customer]))
    locations = list(set([c.location for c in combinations if c.location]))
    
    return {
        'products': sorted(products),
        'customers': sorted(customers),
        'locations': sorted(locations)
    }

def get_forecast_data_with_combinations(
    db: Session,
    product: Optional[str] = None,
    customer: Optional[str] = None,
    location: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 50
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Get forecast data with combination details using optimized joins
    """
    query = db.query(ForecastData, ProductCustomerLocationCombination)\
        .join(ProductCustomerLocationCombination, ForecastData.combination_id == ProductCustomerLocationCombination.id)
    
    # Apply filters
    if product:
        query = query.filter(ProductCustomerLocationCombination.product == product)
    if customer:
        query = query.filter(ProductCustomerLocationCombination.customer == customer)
    if location:
        query = query.filter(ProductCustomerLocationCombination.location == location)
    if start_date:
        query = query.filter(ForecastData.date >= start_date)
    if end_date:
        query = query.filter(ForecastData.date <= end_date)
    
    # Get total count
    total_count = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    results = query.offset(offset).limit(page_size).all()
    
    # Format results
    formatted_results = []
    for forecast_data, combination in results:
        formatted_results.append({
            'id': forecast_data.id,
            'product': combination.product,
            'customer': combination.customer,
            'location': combination.location,
            'product_group': combination.product_group,
            'product_hierarchy': combination.product_hierarchy,
            'location_region': combination.location_region,
            'customer_group': combination.customer_group,
            'customer_region': combination.customer_region,
            'ship_to_party': combination.ship_to_party,
            'sold_to_party': combination.sold_to_party,
            'quantity': float(forecast_data.quantity),
            'uom': forecast_data.uom,
            'date': forecast_data.date.isoformat(),
            'unit_price': float(forecast_data.unit_price) if forecast_data.unit_price else None,
            'created_at': forecast_data.created_at.isoformat(),
            'updated_at': forecast_data.updated_at.isoformat()
        })
    
    return formatted_results, total_count

def get_aggregated_data_optimized(
    db: Session,
    forecast_by: str,
    selected_item: Optional[str] = None,
    selected_product: Optional[str] = None,
    selected_customer: Optional[str] = None,
    selected_location: Optional[str] = None,
    selected_products: Optional[List[str]] = None,
    selected_customers: Optional[List[str]] = None,
    selected_locations: Optional[List[str]] = None,
    selected_items: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get aggregated forecast data using optimized queries with combination table
    """
    from sqlalchemy import func
    
    query = db.query(
        ForecastData.date,
        func.sum(ForecastData.quantity).label('total_quantity'),
        ProductCustomerLocationCombination.product,
        ProductCustomerLocationCombination.customer,
        ProductCustomerLocationCombination.location
    ).join(ProductCustomerLocationCombination, ForecastData.combination_id == ProductCustomerLocationCombination.id)
    
    # Apply filters based on forecast mode
    if selected_products:
        query = query.filter(ProductCustomerLocationCombination.product.in_(selected_products))
    if selected_customers:
        query = query.filter(ProductCustomerLocationCombination.customer.in_(selected_customers))
    if selected_locations:
        query = query.filter(ProductCustomerLocationCombination.location.in_(selected_locations))
    if selected_items:
        if forecast_by == 'product':
            query = query.filter(ProductCustomerLocationCombination.product.in_(selected_items))
        elif forecast_by == 'customer':
            query = query.filter(ProductCustomerLocationCombination.customer.in_(selected_items))
        elif forecast_by == 'location':
            query = query.filter(ProductCustomerLocationCombination.location.in_(selected_items))
    
    # Single item filters
    if selected_item:
        if forecast_by == 'product':
            query = query.filter(ProductCustomerLocationCombination.product == selected_item)
        elif forecast_by == 'customer':
            query = query.filter(ProductCustomerLocationCombination.customer == selected_item)
        elif forecast_by == 'location':
            query = query.filter(ProductCustomerLocationCombination.location == selected_item)
    
    # Precise combination filters
    if selected_product:
        query = query.filter(ProductCustomerLocationCombination.product == selected_product)
    if selected_customer:
        query = query.filter(ProductCustomerLocationCombination.customer == selected_customer)
    if selected_location:
        query = query.filter(ProductCustomerLocationCombination.location == selected_location)
    
    # Group by date and combination
    query = query.group_by(
        ForecastData.date,
        ProductCustomerLocationCombination.product,
        ProductCustomerLocationCombination.customer,
        ProductCustomerLocationCombination.location
    ).order_by(ForecastData.date)
    
    results = query.all()
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            'date': result.date.isoformat(),
            'quantity': float(result.total_quantity),
            'product': result.product,
            'customer': result.customer,
            'location': result.location
        })
    
    return formatted_results