using System;
using System.Drawing;

namespace WebApplication1
{
    public partial class StudentRegistration : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
        }

        protected void btnSubmit_Click(object sender, EventArgs e)
        {
            if (!string.IsNullOrWhiteSpace(txtRollNo.Text) && !string.IsNullOrWhiteSpace(txtName.Text))
            {
                lblMessage.ForeColor = Color.Green;
                lblMessage.Text = "Success! Student " + txtName.Text + " (" + txtRollNo.Text + ") registered for " + ddlBranch.SelectedValue + ".";
                txtRollNo.Text = "";
                txtName.Text = "";
                txtEmail.Text = "";
            }
            else
            {
                lblMessage.ForeColor = Color.Red;
                lblMessage.Text = "Please enter both Roll Number and Name.";
            }
        }
    }
}