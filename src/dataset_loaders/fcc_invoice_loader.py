from __future__ import annotations
from .base_loader import BaseDatasetLoader
class FCCInvoiceLoader(BaseDatasetLoader):
    dataset_name = 'fcc_invoices'
    default_label_map = {
        'Line Item - Start Data': 'Line Item - Start Date',
        'line item - start data': 'Line Item - Start Date',
        'Agency': 'Agency', 'Advertiser': 'Advertiser',
        'Line Item - Description': 'Line Item - Description',
        'Line Item - Rate': 'Line Item - Rate',
        'Line Item - Start Date': 'Line Item - Start Date',
        'Line Item - End Date': 'Line Item - End Date',
        'Line Item - Days': 'Line Item - Days',
    }
