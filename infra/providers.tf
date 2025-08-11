terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.116"
    }
  }

  # backend "azurerm" {}  # (optional) if you use remote state
}

provider "azurerm" {
  features {}
}
