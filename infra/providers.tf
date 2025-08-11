terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.116"
    }
  }
  # Optional: remote state in Azure Storage
  # backend "azurerm" {}
}

provider "azurerm" { features {} }
